"""TSP instance files and their rasterization.

One instance per line:
    x1 y1 x2 y2 ... output t1 t2 ... t1 [output c1 c2 ...]
with 1-based reference tours and, for the constrained benchmark, a constraint
block (box: y_bottom y_top x_left x_right; path: pairs of 0-based city indices;
cluster: one label per city). See data/README.md.

`draw_tour` and `TSPDataset` extend `rasterize_tsp` and `TSPDataset` of
diffusion_priors (Graikos et al., NeurIPS 2022;
https://github.com/AlexGraikos/diffusion_priors, MIT License,
tsp/tsp_utils.py and tsp/inference.py) with the constraint drawing.
"""
import cv2
import numpy as np
import torch

# gray level per cluster label, used only when drawing a cluster instance
_CLUSTER_COLORS = {0: 0.9, 1: 0.1, 2: 0.8, 3: 0.2, 4: 0.7, 5: 0.3, 6: 0.6, 7: 0.4}


def draw_tour(tour, points, box=None, paths=None, cluster=None, show_constraint=False,
              img_size=64, line_color=0.5, line_thickness=2, point_color=1, point_circle=True,
              point_radius=2, show_position=False, box_color=0.75):
    """Rasterize a tour (0- or 1-based) and optional constraints; returns an image in [-1, 1]."""
    img = np.zeros((img_size, img_size))

    # tour edges
    for i in range(tour.shape[0] - 1):
        if tour.min() == 1:
            from_idx, to_idx = tour[i] - 1, tour[i + 1] - 1
        elif tour.min() == 0:
            from_idx, to_idx = tour[i], tour[i + 1]
        cv2.line(img,
                 tuple(((img_size - 1) * points[from_idx, ::-1]).astype(int)),
                 tuple(((img_size - 1) * points[to_idx, ::-1]).astype(int)),
                 color=line_color, thickness=line_thickness)

    # cities
    for i in range(points.shape[0]):
        point_coords = tuple(((img_size - 1) * points[i, ::-1]).astype(int))
        color = _CLUSTER_COLORS[cluster[i]] if cluster is not None else point_color
        if point_circle:
            cv2.circle(img, point_coords, radius=point_radius, color=color, thickness=-1)
        else:
            row = round((img_size - 1) * points[i, 0])
            col = round((img_size - 1) * points[i, 1])
            img[row, col] = point_color
        if show_position:
            text = f'{i}_({points[i, 1]:.2f}, {points[i, 0]:.2f})'
            cv2.putText(img, text, (point_coords[0] + 5, point_coords[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv2.LINE_AA)

    # obstacle
    if box is not None:
        y_bottom = int(box[0] * (img_size - 1))
        y_top = int(box[1] * (img_size - 1))
        x_left = int(box[2] * (img_size - 1))
        x_right = int(box[3] * (img_size - 1))
        img[y_bottom:y_top, x_left:x_right] = box_color

    # mandatory edges
    if paths is not None:
        for i in range(0, len(paths), 2):
            from_idx, to_idx = int(paths[i]), int(paths[i + 1])
            p_from = tuple(((img_size - 1) * points[from_idx, ::-1]).astype(int))
            p_to = tuple(((img_size - 1) * points[to_idx, ::-1]).astype(int))
            cv2.line(img, p_from, p_to, line_color, thickness=line_thickness)
            if show_constraint:
                cv2.line(img, p_from, p_to, box_color, thickness=line_thickness)

    return 2 * (img - 0.5)


class TSPDataset(torch.utils.data.Dataset):
    """Yields (image, points, reference tour, index, constraint) per instance.

    The image is the reference tour drawn on the canvas; DDRL uses it only for
    its shape (x_T) and for the city pixels (value 1) that f(phi) is overlaid with.
    """

    def __init__(self, data_file, img_size, constraint_type='basic', point_radius=1, point_color=1,
                 point_circle=True, line_thickness=2, line_color=0.5, box_color=0.75):
        self.data_file = data_file
        self.img_size = img_size
        self.point_radius = point_radius
        self.point_color = point_color
        self.point_circle = point_circle
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.box_color = box_color
        self.constraint_type = constraint_type

        with open(data_file) as f:
            self.file_lines = f.read().splitlines()
        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

    def __len__(self):
        return len(self.file_lines)

    def parse(self, idx):
        """Return (points (N, 2), 1-based tour, constraint or None) of instance idx."""
        parts = self.file_lines[idx].strip().split(' output ')
        coords = parts[0].split(' ')
        points = np.array([[float(coords[i]), float(coords[i + 1])] for i in range(0, len(coords), 2)])
        tour = np.array([int(t) for t in parts[1].split(' ')])
        constraint = None
        if self.constraint_type != 'basic':
            constraint = np.array([float(t) for t in parts[2].split(' ')])
        return points, tour, constraint

    def rasterize(self, idx):
        points, tour, constraint = self.parse(idx)
        img = draw_tour(tour=tour, points=points,
                        box=constraint if self.constraint_type == 'box' else None,
                        img_size=self.img_size, line_color=self.line_color,
                        line_thickness=self.line_thickness, point_color=self.point_color,
                        point_circle=self.point_circle, point_radius=self.point_radius,
                        box_color=self.box_color)
        return img, points, tour, constraint

    def __getitem__(self, idx):
        img, points, tour, constraint = self.rasterize(idx)
        if self.constraint_type == 'basic':
            constraint = 0  # placeholder so the default collate works
        return img[np.newaxis, :, :], points, tour, idx, constraint
