"""The latent phi, its adjacency matrix A(phi) and its edge image f(phi).

phi in R^{N x N} is the only learnable parameter of DDRL (Sec. 3.1). Paper,
Sec. 3.3 ("Sampling Phase"):

    A_ij(phi) = exp(phi_ij) / sum_{m != i} exp(phi_im),   A_ii(phi) = 0
    f(phi)    = sum_{i,j} A_ij(phi) * E_ij

where E_ij is the rasterized image of edge (i, j). f(phi) takes the place of the
clean-image estimate x_0 inside the DDIM step (Eq. 1).

`LatentTourModel` starts from the `InferenceModel` of diffusion_priors
(Graikos et al., NeurIPS 2022; https://github.com/AlexGraikos/diffusion_priors,
MIT License, tsp/inference.py), as does `adjacency_from_latent` (`normalize`
there). DDRL adds the heatmap (logit) parameterization, in-place
re-initialization, and a chunked f(phi) for large N.
"""
import os

import cv2
import numpy as np
import torch
from torch import nn

# f(phi) multiplies an (N*N, H, W) stack of edge images by A(phi). Above this many
# elements (2^31, 8 GiB in float32; first exceeded at N = 500 with a 128 px canvas)
# the sum is accumulated in slices of _ENCODE_CHUNK_ELEMS elements instead. Below
# the threshold the single fused product is used, which is the computation the
# paper's numbers at N <= 200 were produced with (slicing reorders the float sum).
# Setting DDRL_ENCODE_ELEM_BUDGET=0 forces the sliced path, e.g. to compare the two.
_ENCODE_ELEM_BUDGET = int(os.environ.get('DDRL_ENCODE_ELEM_BUDGET', 2 ** 31))
_ENCODE_CHUNK_ELEMS = int(os.environ.get('DDRL_ENCODE_CHUNK_ELEMS', 2 ** 28))


def adjacency_from_latent(latent, mode, entropy_reg=0.1, eps=1e-6):
    """Row-wise softmax with the diagonal masked: A(phi) of the paper.

    mode='logit': phi is the latent itself (used when initializing from a DIFUSCO
        heatmap, phi = log(h / (1 - h))).
    mode='cost':  the stored latent L is a cost-like matrix, as in diffusion_priors,
        and phi = -entropy_reg * L (used for the prior and random initializations).

    The diagonal is masked by subtracting 1e5 (A_ii ~ 0), and eps is added to every
    entry. Returns a (B, N, N) tensor.
    """
    if mode == 'logit':
        phi = latent
    elif mode == 'cost':
        phi = -entropy_reg * latent
    else:
        raise ValueError(f"Unknown mode: {mode}")

    phi = phi - torch.eye(phi.shape[-1], device=phi.device) * 100000
    log_a = phi - torch.logsumexp(phi, dim=-1, keepdim=True)
    return torch.exp(log_a) + eps


class LatentTourModel(nn.Module):
    """Holds phi and maps it to A(phi) and f(phi).

    Args:
        batch_size: number of latents optimized jointly (1 in the paper).
        num_points: N.
        img_size: canvas size H = W of the edge images.
        line_color, line_thickness: rasterization of an edge (as in the prior's data).
        xT: fixed initial noise x_T of every denoising trajectory.
        use_heatmap: True for DDRL-DIFUSCO (logit parameterization), False otherwise.
    """

    def __init__(self, batch_size, num_points, img_size, line_color, line_thickness, xT, use_heatmap):
        super().__init__()
        self.num_points = num_points
        self.batch_size = batch_size
        self.img_size = img_size
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.xT = xT
        self.mode = 'logit' if use_heatmap else 'cost'

        self.latent = nn.Parameter(torch.randn(batch_size, num_points, num_points))
        self.adj_mat = adjacency_from_latent(self.latent, self.mode)

    # ---------------------------------------------------------- initialization
    def reset(self):
        """Redraw the latent from N(0, I) in place (the optimizer keeps tracking it)."""
        nn.init.normal_(self.latent)

    def init_from_heatmap(self, heatmap, device):
        """DDRL-DIFUSCO initialization: phi = log(h / (1 - h)), h clipped to [1e-6, 1 - 1e-6]
        (Supp. Sec. B, "Initialization from a DIFUSCO heatmap")."""
        heatmap = np.clip(heatmap, 1e-6, 1 - 1e-6)
        logit = np.log(heatmap / (1 - heatmap))
        init = torch.tensor(logit, dtype=torch.float32, device=device)
        init = init.reshape(1, self.num_points, self.num_points).repeat(self.batch_size, 1, 1)
        # Write in place: the optimizer was built from model.parameters(), so a new
        # nn.Parameter would leave it stepping a tensor that is no longer used.
        with torch.no_grad():
            self.latent.data = self.latent.data.to(device)
            self.latent.data.copy_(init)
        self.latent.requires_grad = True
        self.adj_mat = adjacency_from_latent(self.latent, self.mode)

    # ---------------------------------------------------------- A(phi), f(phi)
    def get_adj_mat(self):
        """A(phi), shape (B, N, N)."""
        return adjacency_from_latent(self.latent, self.mode)

    def compute_edge_images(self, points, img_query):
        """Rasterize every edge image E_ij once per instance.

        img_query marks the city pixels of the input image; they are drawn on top
        of f(phi) so that the image always shows the cities, as in the prior's
        training images.
        """
        self.img_query = img_query
        num_nodes = points.shape[0]
        self.edge_images = torch.zeros((num_nodes, num_nodes, self.img_size, self.img_size),
                                       dtype=torch.float32, device=self.latent.device)
        scaled_points = ((self.img_size - 1) * points[:, ::-1]).astype(int)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                edge_img = np.zeros((self.img_size, self.img_size), dtype=np.float16)
                cv2.line(edge_img, tuple(scaled_points[i]), tuple(scaled_points[j]),
                         color=self.line_color, thickness=self.line_thickness)
                edge_tensor = torch.from_numpy(edge_img).float().to(self.latent.device)
                self.edge_images[i, j] = edge_tensor
                self.edge_images[j, i] = edge_tensor

    def encode(self):
        """f(phi) = sum_ij A_ij(phi) E_ij, rescaled to [-1, 1], with the cities drawn on top.

        Returns a (B, 1, H, W) tensor; differentiable in phi.
        """
        self.adj_mat = adjacency_from_latent(self.latent, self.mode)
        all_edges = self.edge_images.view(1, -1, self.img_size, self.img_size).to(self.adj_mat.device)
        if all_edges.numel() <= _ENCODE_ELEM_BUDGET:
            img = all_edges * self.adj_mat.view(self.batch_size, -1, 1, 1)  # (B, N*N, H, W)
            img = torch.sum(img, dim=1, keepdims=True)                       # (B, 1, H, W)
        else:
            w = self.adj_mat.view(self.batch_size, -1)
            img = torch.zeros(self.batch_size, 1, self.img_size, self.img_size,
                              device=self.adj_mat.device, dtype=all_edges.dtype)
            step = max(1, _ENCODE_CHUNK_ELEMS // (self.img_size * self.img_size))
            for s in range(0, all_edges.shape[1], step):
                e = s + step
                img = img + torch.sum(all_edges[:, s:e] * w[:, s:e].view(self.batch_size, -1, 1, 1),
                                      dim=1, keepdims=True)

        img = 2 * (img - 0.5)  # same [-1, 1] range as the prior's training images
        img[self.img_query.tile(self.batch_size, 1, 1, 1) == 1] = 1
        return img
