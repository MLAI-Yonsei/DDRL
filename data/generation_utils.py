"""Helpers of the benchmark generation scripts (make_dataset_{box,path,cluster}.py).

Reference tours are computed with Concorde on a modified distance matrix
(Supp. Sec. C, "Dataset Generation"): edges that cross the obstacle or leave the
required path structure get a penalty of 100, and the clustered TSP adds the
same penalty to inter-cluster edges.
"""
import random

import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull

from ddrl.geometry import do_intersect, does_intersect_box


def write_tsplib_file(distance_matrix, filename, scale_factor=1000):
    """Write an explicit full distance matrix in TSPLIB format (integer weights)."""
    size = len(distance_matrix)
    with open(filename, 'w') as f:
        f.write("NAME: TSP\nTYPE: TSP\nDIMENSION: {}\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
                "EDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n".format(size))
        for row in distance_matrix:
            f.write(" ".join(str(int(val * scale_factor)) for val in row) + "\n")
        f.write("EOF\n")


def calculate_distance_matrix(points, edges=None, box=None):
    """Euclidean distances with a +100 penalty for edges crossing `box` and, if
    `edges` is given, for every edge not in that list."""
    num_points = points.shape[0]
    distance_matrix = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            if box is not None and does_intersect_box(points[i], points[j], box):
                distance += 100
            if edges is not None and ([i, j] not in edges and [j, i] not in edges):
                distance += 100
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    return distance_matrix


def check_tour_intersections(tour, points):
    """True if two non-adjacent edges of the (1-based) tour cross."""
    for i in range(len(tour) - 1):
        p1, q1 = points[tour[i] - 1], points[tour[i + 1] - 1]
        for j in range(i + 2, len(tour) - 1):
            if i == 0 and j == len(tour) - 2:
                continue  # first and last edges share a city
            p2, q2 = points[tour[j] - 1], points[tour[j + 1] - 1]
            if do_intersect(p1, q1, p2, q2):
                return True
    return False


# ----------------------------------------------------------------- obstacle

def check_tour_box_overlap(tour, box, points):
    """True if an edge of the (1-based) tour crosses the box."""
    for i in range(len(tour) - 1):
        if does_intersect_box(points[tour[i] - 1], points[tour[i + 1] - 1], box):
            return True
    return False


def calculate_intersection_and_overlap(x_left, x_right, y_bottom, y_top, points, gt_tour):
    """Number of reference-tour edges crossing the box, and a rough overlap measure."""
    intersection, overlap = 0, 0
    box_edges = [
        ((x_left, y_bottom), (x_left, y_top)),
        ((x_left, y_top), (x_right, y_top)),
        ((x_right, y_top), (x_right, y_bottom)),
        ((x_right, y_bottom), (x_left, y_bottom)),
    ]
    for i in range(len(gt_tour) - 1):
        p1, q1 = points[gt_tour[i] - 1], points[gt_tour[i + 1] - 1]
        if any(do_intersect(p1, q1, p3, q3) for p3, q3 in box_edges):
            intersection += 1
            overlap += (min(x_right, q1[0]) - max(x_left, p1[0])) * (min(y_top, q1[1]) - max(y_bottom, p1[1]))
    return intersection, overlap


def is_valid_box(x_left, x_right, y_bottom, y_top, points):
    """A box is valid if it contains no city."""
    return all(not (x_left < px < x_right and y_bottom < py < y_top) for px, py in points)


def find_optimal_box(points, gt_tour):
    """Grid search for the empty box inside the convex hull that the unconstrained
    reference tour crosses most (ties: larger overlap, then smaller area)."""
    hull_path = Path(points[ConvexHull(points).vertices])
    MIN_EDGE_DISTANCE = 0.05
    best_coords, max_intersection, max_overlap, max_area = None, 0, 0, float('inf')
    x_start, x_end = points[:, 0].min(), points[:, 0].max()
    y_start, y_end = points[:, 1].min(), points[:, 1].max()

    for y_bottom in np.arange(y_start + MIN_EDGE_DISTANCE, y_end - MIN_EDGE_DISTANCE, 0.1):
        for x_left in np.arange(x_start + MIN_EDGE_DISTANCE, x_end - MIN_EDGE_DISTANCE, 0.1):
            for y_top in np.arange(y_bottom + 0.1, y_end - MIN_EDGE_DISTANCE, 0.1):
                for x_right in np.arange(x_left + 0.1, x_end - MIN_EDGE_DISTANCE, 0.1):
                    corners = np.array([[x_left, y_bottom], [x_left, y_top], [x_right, y_bottom], [x_right, y_top]])
                    if not np.all(hull_path.contains_points(corners)) or y_top <= y_bottom or x_right <= x_left:
                        continue
                    if not is_valid_box(x_left, x_right, y_bottom, y_top, points):
                        continue
                    intersection, overlap = calculate_intersection_and_overlap(
                        x_left, x_right, y_bottom, y_top, points, gt_tour)
                    area = (y_top - y_bottom) * (x_right - x_left)
                    if (intersection > max_intersection
                            or (intersection == max_intersection and overlap > max_overlap)
                            or (intersection == max_intersection and overlap == max_overlap and area < max_area)):
                        best_coords = (x_left, x_right, y_bottom, y_top)
                        max_intersection, max_overlap, max_area = intersection, overlap, area
    if best_coords:
        print(f"Best coordinates: top-left ({best_coords[0]}, {best_coords[2]}), "
              f"bottom-right ({best_coords[1]}, {best_coords[3]})")
    else:
        print("No valid rectangle found")
    return best_coords


def generate_random_box(gt_tour, points, width_max=0.2, height_max=0.2):
    """Fallback: a random empty box centred on a reference-tour edge."""
    def empty(box):
        x_left, x_right, y_bottom, y_top = box
        return all(not (x_left <= x <= x_right and y_bottom <= y <= y_top) for x, y in points)

    for _ in range(1000):
        idx = random.choice(range(len(gt_tour) - 1))
        p1, p2 = points[gt_tour[idx] - 1], points[gt_tour[idx + 1] - 1]
        cx, cy = (p1 + p2) / 2
        width = random.uniform(0.05, width_max)
        height = random.uniform(0.05, height_max)
        box = (cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2)
        if empty(box):
            return box


# ----------------------------------------------------------------- path

def check_edge_intersection(new_edge, existing_edges, points):
    p1, q1 = points[new_edge[0]], points[new_edge[1]]
    for edge in existing_edges:
        if do_intersect(p1, q1, points[edge[0]], points[edge[1]]):
            return True
    return False


def sampling_edge(gt_tour, points, sample_cnt=1):
    """Sample `sample_cnt` mutually non-crossing city pairs that are not adjacent in
    the reference tour (0-based, [min, max])."""
    n = len(gt_tour)
    edges = []
    while len(edges) < sample_cnt:
        while True:
            idx1, idx2 = random.sample(range(n), 2)
            if abs(idx1 - idx2) != 1 and idx1 != idx2:
                break
        num1, num2 = gt_tour[idx1] - 1, gt_tour[idx2] - 1
        new_edge = [min(num1, num2), max(num1, num2)]
        if new_edge not in edges and not check_edge_intersection(new_edge, edges, points):
            edges.append(new_edge)
    return edges


# ----------------------------------------------------------------- cluster

def adjust_distances_for_clusters(distance_matrix, labels, penalty=100):
    """Add `penalty` to every inter-cluster edge (clustered-TSP transformation)."""
    adjusted = distance_matrix.copy()
    n = distance_matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                adjusted[i, j] += penalty
                adjusted[j, i] += penalty
    return adjusted
