"""Generate the obstacle-constrained ("box") TSP benchmark from a basic test set.

For each instance an empty rectangle is placed where the unconstrained reference
tour crosses it most (`find_optimal_box`; a random empty box centred on a tour
edge if none is found), and Concorde solves the TSP with a +100 penalty on edges
that cross it. A tour is kept only if it avoids the box and does not cross
itself (Supp. Sec. C, "Obstacle Constraint Dataset Generation").

Usage (from the repository root, pyconcorde installed):
    python data/make_dataset_box.py --num_cities 20
reads data/tsp20_test_concorde.txt and writes data/tsp20_box_constraint_<yymmdd>.txt.
The released files are the benchmark of record; this script does not reproduce
them bit for bit (see data/README.md).
"""
import argparse
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np
from concorde.tsp import TSPSolver
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddrl.dataset import TSPDataset  # noqa: E402
from ddrl.utils import seed_everything  # noqa: E402
from generation_utils import (calculate_distance_matrix, check_tour_box_overlap,  # noqa: E402
                              check_tour_intersections, find_optimal_box,
                              generate_random_box, write_tsplib_file)


def solve_with_box(points, box, problem_path):
    """Concorde tour that avoids `box`, or None."""
    write_tsplib_file(calculate_distance_matrix(points, box=box), problem_path)
    solution = TSPSolver.from_tspfile(problem_path).solve()
    if solution is None:
        return None
    route = np.append(solution.tour, solution.tour[0]) + 1
    if check_tour_box_overlap(route, box, points) or check_tour_intersections(route, points):
        return None
    return route


def main():
    parser = argparse.ArgumentParser(description='Generate the obstacle-constrained TSP benchmark.')
    parser.add_argument('--num_cities', default=20, type=int)
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--work_dir', default='./problem', help='scratch directory for TSPLIB files')
    args = parser.parse_args()

    seed_everything(2024)
    os.makedirs(args.work_dir, exist_ok=True)
    problem_path = os.path.join(args.work_dir, 'tsp_problem.tsp')
    in_path = os.path.join(args.data_path, f'tsp{args.num_cities}_test_concorde.txt')
    out_path = os.path.join(args.data_path,
                            f'tsp{args.num_cities}_box_constraint_{datetime.today().strftime("%y%m%d")}.txt')
    dataset = TSPDataset(data_file=in_path, img_size=64, constraint_type='basic')

    with open(out_path, 'w') as f:
        for i in tqdm(range(len(dataset))):
            points, gt_tour, _ = dataset.parse(i)

            box = find_optimal_box(points, gt_tour)
            route = solve_with_box(points, box, problem_path) if box else None
            while route is None:
                box = generate_random_box(gt_tour, points)
                route = solve_with_box(points, box, problem_path)

            str_points = ' '.join(str(v) for v in points.flatten().tolist())
            str_tour = ' '.join(str(v) for v in route.flatten().tolist())
            str_box = ' '.join(str(v) for v in box)
            f.write(f'{str_points} output {str_tour} output {str_box} \n')

            for leftover in glob('*.res') + glob('*.sol'):  # Concorde side files
                os.remove(leftover)


if __name__ == '__main__':
    main()
