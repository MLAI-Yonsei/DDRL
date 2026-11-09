"""Decoding an adjacency matrix A(phi) into a tour (the graph side of the reward).

Two decoders share the rule of the paper (Sec. 3.3, "Sampling Phase"): edges are
inserted greedily in decreasing order of A_ij / d_ij, so likely and short edges
enter first, until every city has degree two and the edges form one cycle.

* `merge_tour` (basic TSP) runs the greedy merge of DIFUSCO in Cython and refines
  the tour with DIFUSCO's batched 2-opt on the GPU.
* `greedy_constrained_tour` (obstacle, path and cluster constraints) is the same
  greedy insertion with the per-constraint admission rules of Supplementary
  Algorithm A4 ("Constrained TSP Tour Construction"); its 2-opt is the
  lexicographic one in ddrl/solver.py.
"""
import hashlib
import random
import warnings

import numpy as np
import torch

from ddrl.cython_merge.cython_merge import merge_cython
from ddrl.geometry import segments_intersect_any


# ---------------------------------------------------------------------------
# Basic TSP: greedy merge + batched 2-opt
#
# `cython_merge`, `merge_tour` and `batched_two_opt_torch` are adapted from
# DIFUSCO (Sun & Yang, NeurIPS 2023; https://github.com/Edward-Sun/DIFUSCO,
# MIT License, difusco/utils/tsp_utils.py). `merge_tour` is DIFUSCO's
# `merge_tours` reduced to one dense adjacency matrix and one sample.
# ---------------------------------------------------------------------------

def cython_merge(points, adj_mat):
    """Greedy edge merge (Cython, from DIFUSCO). Returns the 0/1 tour adjacency matrix."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_adj_mat, merge_iterations = merge_cython(points.astype("double"), adj_mat.astype("double"))
        real_adj_mat = np.asarray(real_adj_mat)
    return real_adj_mat, merge_iterations


def merge_tour(adj_mat, points):
    """Decode a (1, N, N) adjacency matrix into a closed tour [0, ..., 0]."""
    sym_adj_mat = adj_mat[0] + adj_mat[0].T
    real_adj_mat, _ = cython_merge(points, sym_adj_mat)

    # walk the degree-2 graph starting from city 0
    tour = [0]
    while len(tour) < sym_adj_mat.shape[0] + 1:
        n = np.nonzero(real_adj_mat[tour[-1]])[0]
        if len(tour) > 1:
            n = n[n != tour[-2]]
        tour.append(n.max())
    return tour


def batched_two_opt_torch(points, tour, max_iterations=1000, device="cpu"):
    """Best-improvement 2-opt on the GPU (from DIFUSCO).

    Each iteration evaluates every 2-opt move at once and applies the best one;
    it stops when no move shortens the tour by more than 1e-6 or after
    `max_iterations` moves.
    """
    iterator = 0
    tour = tour.copy()

    with torch.inference_mode():
        cuda_points = torch.from_numpy(points).to(device)
        cuda_tour = torch.from_numpy(tour).to(device)
        batch_size = cuda_tour.shape[0]
        min_change = -1.0

        while min_change < 0.0:
            points_i = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j = cuda_points[cuda_tour[:, :-1].reshape(-1)].reshape((batch_size, 1, -1, 2))
            points_i_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, -1, 1, 2))
            points_j_plus_1 = cuda_points[cuda_tour[:, 1:].reshape(-1)].reshape((batch_size, 1, -1, 2))

            A_ij = torch.sqrt(torch.sum((points_i - points_j) ** 2, axis=-1))
            A_i_plus_1_j_plus_1 = torch.sqrt(torch.sum((points_i_plus_1 - points_j_plus_1) ** 2, axis=-1))
            A_i_i_plus_1 = torch.sqrt(torch.sum((points_i - points_i_plus_1) ** 2, axis=-1))
            A_j_j_plus_1 = torch.sqrt(torch.sum((points_j - points_j_plus_1) ** 2, axis=-1))

            change = A_ij + A_i_plus_1_j_plus_1 - A_i_i_plus_1 - A_j_j_plus_1
            valid_change = torch.triu(change, diagonal=2)

            min_change = torch.min(valid_change)
            flatten_argmin_index = torch.argmin(valid_change.reshape(batch_size, -1), dim=-1)
            min_i = torch.div(flatten_argmin_index, len(points), rounding_mode='floor')
            min_j = torch.remainder(flatten_argmin_index, len(points))

            if min_change < -1e-6:
                for i in range(batch_size):
                    cuda_tour[i, min_i[i] + 1:min_j[i] + 1] = torch.flip(
                        cuda_tour[i, min_i[i] + 1:min_j[i] + 1], dims=(0,))
                iterator += 1
            else:
                break

            if iterator >= max_iterations:
                break

        tour = cuda_tour.cpu().numpy()
    return tour, iterator


# ---------------------------------------------------------------------------
# Constrained TSP: greedy insertion with admission rules (Supp. Algorithm A4)
#
# The greedy insertion loop and the component bookkeeping start from the
# decoder of diffusion_priors (Graikos et al., NeurIPS 2022;
# https://github.com/AlexGraikos/diffusion_priors, MIT License, tsp/inference.py).
# The constraint rules, the non-crossing test, the dead-end fallback and the
# vectorized bookkeeping are DDRL's.
# ---------------------------------------------------------------------------

def _seed_from_inputs(points, adj_mat):
    """Seed for the dead-end tie-break of the constrained walk.

    Derived from the inputs, so the same instance and adjacency give the same
    tour, while different inputs keep different random choices.
    """
    h = hashlib.md5(np.ascontiguousarray(points).tobytes())
    h.update(np.ascontiguousarray(adj_mat).tobytes())
    return int.from_bytes(h.digest()[:8], 'little')


def greedy_constrained_tour(adj_mat, points, dists, constraint_type='basic', constraint=None):
    """Decode an (N, N) adjacency matrix into a closed tour that respects the constraint.

    Admission rules (Supp. Sec. E, "Constrained Decoding"):
      * every type: an edge is skipped if it closes a sub-cycle, touches a city
        that already has degree two, or crosses an accepted edge;
      * obstacle ('box'):  `constraint` is the 0/1 crossing matrix, and an edge
        that crosses the box is skipped;
      * 'path':  `constraint` is a flat list of mandatory pairs, inserted first;
      * 'cluster': `constraint` gives the cluster label per city; an
        inter-cluster edge is admitted only while both clusters have fewer than
        two inter-cluster edges, so every cluster is entered and left once.

    Note: `adj_mat` is symmetrized in place (W = A + A^T in Algorithm A4).

    Bookkeeping: `endpoint[x]` is True while x is the end of a partial chain and
    `partner[x]` is the other end of that chain, so "would this edge close a
    sub-cycle" is `partner[a] == b`. Accepted edges with a < b are kept in
    `edge_p1/edge_p2` for the vectorized crossing test. Edges stored with a > b
    are deliberately not tested for crossings; this reproduces the decoder the
    paper's numbers were measured with.
    """
    adj_mat += adj_mat.T
    num_cities = adj_mat.shape[0]
    real_adj_mat = np.zeros_like(adj_mat)

    np.seterr(divide='ignore', invalid='ignore')

    endpoint = [True] * num_cities
    partner = list(range(num_cities))
    num_components = num_cities
    final_row = None  # (start, end) of the last merged chain, used to close the cycle

    max_edges = num_cities + (len(constraint) if constraint_type == 'path' and constraint is not None else 0) + 2
    edge_p1 = np.empty((max_edges, 2), dtype=points.dtype)
    edge_p2 = np.empty((max_edges, 2), dtype=points.dtype)
    n_edges = 0
    # touched[x]: x is an endpoint of a tested edge. A candidate that shares a city
    # with a tested edge always "intersects" it under do_intersect (the shared point
    # is collinear and on the segment), so such candidates are rejected directly.
    touched = [False] * num_cities

    if constraint_type == 'path':
        # mandatory edges enter before the greedy loop
        path = constraint
        for i in range(0, len(path), 2):
            a, b = int(path[i]), int(path[i + 1])
            if a == b:
                continue  # degenerate pair in the data: unsatisfiable, treated as no-op
            real_adj_mat[a, b] = 1
            if a < b:
                edge_p1[n_edges] = points[a]
                edge_p2[n_edges] = points[b]
                n_edges += 1
                touched[a] = touched[b] = True
            if not (endpoint[a] and endpoint[b]):
                raise IndexError('path constraint endpoint is no longer a chain end')
            pa, pb = partner[a], partner[b]
            if pa == b:
                final_row = (b, a)
                continue
            final_row = (pa, pb)
            endpoint[a] = (pa == a)
            endpoint[b] = (pb == b)
            partner[pa] = pb
            partner[pb] = pa
            num_components -= 1

    if constraint_type == 'cluster':
        cluster = constraint
        inter_edge_count = {}  # cluster label -> accepted inter-cluster edges

    # candidate edges in decreasing order of A_ij / d_ij
    order = (-adj_mat / dists).flatten().argsort()
    a_all = (order // num_cities).tolist()
    b_all = (order % num_cities).tolist()
    for a, b in zip(a_all, b_all):
        if a == b or not (endpoint[a] and endpoint[b]):
            continue
        # no edge may cross an accepted one
        if touched[a] or touched[b]:
            continue
        if n_edges and segments_intersect_any(points[a], points[b], edge_p1[:n_edges], edge_p2[:n_edges]):
            continue

        if constraint_type == 'box' and constraint[a][b] == 1:
            continue

        if constraint_type == 'cluster':
            ca_label, cb_label = cluster[a], cluster[b]
            if ca_label != cb_label and (
                inter_edge_count.get(ca_label, 0) >= 2 or inter_edge_count.get(cb_label, 0) >= 2
            ):
                continue

        if partner[a] == b:  # would close a sub-cycle
            continue
        pa, pb = partner[a], partner[b]
        real_adj_mat[a, b] = 1
        if a < b:
            edge_p1[n_edges] = points[a]
            edge_p2[n_edges] = points[b]
            n_edges += 1
            touched[a] = touched[b] = True
        final_row = (pa, pb)
        endpoint[a] = (pa == a)
        endpoint[b] = (pb == b)
        partner[pa] = pb
        partner[pb] = pa
        num_components -= 1

        if constraint_type == 'cluster' and cluster[a] != cluster[b]:
            inter_edge_count[cluster[a]] = inter_edge_count.get(cluster[a], 0) + 1
            inter_edge_count[cluster[b]] = inter_edge_count.get(cluster[b], 0) + 1

        if num_components == 1:
            break

    if num_components == 1:
        if final_row is None:  # only possible when num_cities == 1
            final_row = (0, 0)
        real_adj_mat[final_row[1], final_row[0]] = 1  # close the Hamiltonian path into a cycle
    real_adj_mat += real_adj_mat.T

    return _walk_constrained(adj_mat, real_adj_mat, dists, points, constraint_type, constraint)


def _walk_constrained(adj_mat, real_adj_mat, dists, points, constraint_type=None, constraint=None):
    """Turn the edge set into a tour, starting at city 0.

    When the greedy stage stopped early (no admissible edge was left), the walk
    falls back to the unvisited city with the largest A_ij / d_ij that is allowed
    (Algorithm A4, "fallback"). If none is allowed, one is drawn at random with a
    seed derived from the inputs.
    """
    if constraint_type == 'path':
        mandatory_paths = set()
        for i in range(0, len(constraint), 2):
            a, b = int(constraint[i]), int(constraint[i + 1])
            if a == b:
                continue
            mandatory_paths.add((a, b))
            mandatory_paths.add((b, a))

    num_nodes = real_adj_mat.shape[0]
    tour = [0]
    visited = set(tour)
    adj_over_dists = adj_mat / dists

    while len(tour) < num_nodes:
        current_node = tour[-1]
        neighbors = np.nonzero(real_adj_mat[current_node])[0]
        next_node = None

        # 1) follow a mandatory edge
        if constraint_type == 'path':
            for neighbor in neighbors:
                if neighbor not in visited and (current_node, neighbor) in mandatory_paths:
                    next_node = neighbor
                    break

        # 2) follow a decoded edge that does not cross the tour so far
        if next_node is None:
            C = D = None
            for neighbor in neighbors:
                if neighbor not in visited:
                    if C is None:
                        t = np.asarray(tour)
                        C = points[t[:-1]]
                        D = points[t[1:]]
                    if not segments_intersect_any(points[current_node], points[neighbor], C, D):
                        next_node = neighbor
                        break

        # 3) fallback: best-ranked unvisited city that the obstacle allows
        if next_node is None:
            remaining_nodes = list(set(range(num_nodes)) - visited)
            sorted_remaining_nodes = sorted(
                remaining_nodes, key=lambda node: adj_over_dists[current_node, node], reverse=True)
            for node in sorted_remaining_nodes:
                if constraint_type != 'box' or constraint[current_node, node] == 0:
                    next_node = node
                    break
            else:
                # dead end: every remaining city crosses the obstacle
                next_node = random.Random(_seed_from_inputs(points, adj_mat)).choice(remaining_nodes)

        visited.add(next_node)
        tour.append(next_node)

    tour.append(0)
    return tour
