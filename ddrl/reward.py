"""Reward r(phi): decode A(phi) into a tour, refine it with 2-opt, and score it.

In the MDP of the paper the reward is sparse (Eq. 2): it is r(phi) at the last
denoising step and 0 elsewhere, so the return of a trajectory is r(phi). For
constrained TSP r subtracts a penalty xi per violated constraint:

    r(phi) = -( l(y) + xi * V(y) ),   y = 2opt(decode(A(phi)))

(Sec. 3.2 and Algorithm 1, `cost(y)`).
"""
import numpy as np

from ddrl.decoding import batched_two_opt_torch, greedy_constrained_tour, merge_tour
from ddrl.solver import TwoOptSolver


def decode_and_score(points, adj_mat, dists, constraint_type='basic', constraint=None,
                     max_iter=1000, penalty_weight=0.0):
    """Evaluate one adjacency matrix.

    Args:
        points: (N, 2) city coordinates.
        adj_mat: (1, N, N) tensor A(phi), on the model's device.
        dists: (N, N) Euclidean distance matrix.
        constraint_type, constraint: see TwoOptSolver.
        max_iter: 2-opt iteration cap.
        penalty_weight: xi in the reward. With 0 the reward is the plain tour
            length; the paper uses xi = 1 (`penalty_const` in the config).

    Returns:
        reward: array([r]) with r = -(l(y) + penalty_weight * V(y)).
        info: dict with 'solved_tour' (1-based closed tour), 'basic_cost' l(y)
            and 'penalty_count' V(y).
    """
    if constraint_type == 'basic':
        tour = merge_tour(adj_mat.detach().cpu().numpy(), points)
        solved_tours, _ = batched_two_opt_torch(
            points.astype("float64"), np.array([tour]).astype('int64'),
            max_iterations=max_iter, device=adj_mat.device)
        solved_tour = solved_tours[0]
        solver = TwoOptSolver(points, constraint_type, constraint)
    else:
        tour = greedy_constrained_tour(adj_mat.detach().cpu().numpy()[0], points, dists,
                                       constraint_type=constraint_type, constraint=constraint)
        solver = TwoOptSolver(points, constraint_type, constraint)
        solved_tour, _ = solver.solve_2opt(tour, max_iter)
        solved_tour = solver.repair(solved_tour)  # no-op unless a path/cluster tour is infeasible

    assert solved_tour[-1] == solved_tour[0], 'Tour not a cycle'

    tour_length = solver.evaluate(solved_tour)
    penalty_count = solver.count_constraints(solved_tour)
    reward = -np.array([tour_length + penalty_weight * penalty_count])
    return reward, {
        'solved_tour': np.array(solved_tour) + 1,
        'basic_cost': tour_length,
        'penalty_count': penalty_count,
    }
