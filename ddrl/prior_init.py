"""The diffusion prior eps_prior and the two initializations of phi.

* DDRL-prior: plug-and-play sampling with the pre-trained prior (Supp. Sec. B,
  Eq. A1),
      phi <- phi - lambda * grad_phi || eps - eps_prior( sqrt(alpha_{t_i}) f(phi) + sqrt(1 - alpha_{t_i}) eps, t_i ) ||^2,
  run over a decreasing schedule of t_i.
* DDRL-DIFUSCO: a DIFUSCO heatmap written into phi (LatentTourModel.init_from_heatmap).

`plug_and_play_init` is adapted from the `runlat` routine of diffusion_priors
(Graikos et al., NeurIPS 2022; https://github.com/AlexGraikos/diffusion_priors,
MIT License, tsp/inference.py). Differences: the prior is passed in and kept in
eval mode, and the gradient is cleared after each step instead of before.
"""
import gc
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from prior.diffusion import GaussianDiffusion
from prior.unet import UNetModel


def build_prior(img_size, device, ckpt_dir=None, num_cities=None, ckpt_name=None):
    """U-Net noise predictor eps_prior with the architecture of the released priors.

    If `ckpt_dir` is given, weights are loaded: `ckpt_name` if set, otherwise the
    TSP-50 prior (unet50_64_8.pth) for N <= 200 and the TSP-500 prior
    (unet500_128_8.pth) above. Both checkpoints have the same parameter shapes and
    the U-Net only stores `image_size`, so either loads at any `img_size`.
    The prior stays fixed throughout (it is never trained by DDRL).
    """
    unet = UNetModel(
        image_size=img_size,
        in_channels=1,
        out_channels=1,
        model_channels=64,
        num_res_blocks=2,
        channel_mult=(1, 2, 3, 4),
        attention_resolutions=[16, 8],
        num_heads=4,
    ).to(device)
    ckpt_path = None
    if ckpt_dir is not None:
        if ckpt_name:
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        elif num_cities <= 200:
            ckpt_path = os.path.join(ckpt_dir, 'unet50_64_8.pth')
        else:
            ckpt_path = os.path.join(ckpt_dir, 'unet500_128_8.pth')
        unet.load_state_dict(torch.load(ckpt_path, map_location=device))
    unet.eval()
    return unet, ckpt_path


def plug_and_play_init(model, prior, steps, device, batch_size=1):
    """DDRL-prior initialization of phi (Supp. Eq. A1).

    `steps` gradient steps on the denoising loss, Adam with lr 1 decayed linearly;
    t_i decreases linearly from T = 1000 with a cosine modulation. Uses the
    linear-beta GaussianDiffusion of diffusion_priors (Supp. Sec. A, "Schedule in
    Sampling and Initialization").
    """
    opt = torch.optim.Adam(model.parameters(), lr=1, betas=(0, 0.9))
    lr_schedule = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=1000)
    diffusion = GaussianDiffusion(T=1000, schedule='linear')

    for i in range(steps):
        # t_i: linearly decreasing with a cosine wiggle
        t = ((steps - i) + (steps - i) // 3 * math.cos(i / 50)) / steps * diffusion.T
        t = np.clip(t, 1, diffusion.T)
        t = np.array([t for _ in range(batch_size)]).astype(int)

        # noise f(phi) to x_{t_i} and ask the prior to recover the noise
        xt, epsilon = diffusion.sample(model.encode(), t)
        t = torch.from_numpy(t).float().view(batch_size)
        epsilon_pred = prior(xt.float(), t.to(device))

        loss = F.mse_loss(epsilon_pred, epsilon)
        loss.backward()
        opt.step()
        opt.zero_grad()
        lr_schedule.step()

    gc.collect()
    torch.cuda.empty_cache()


def load_difusco_heatmap(heatmap_dir, num_cities, sample_idx):
    """Heatmap produced by DIFUSCO's test script for one instance (see ckpt/README.md)."""
    return np.load(os.path.join(heatmap_dir, f'tsp{num_cities}', 'numpy_heatmap',
                                f'test-heatmap-{int(sample_idx)}.npy'))
