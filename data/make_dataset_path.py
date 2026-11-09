"""Generate the path-constrained TSP benchmark from a basic test set.

Each instance receives one or two mandatory edges (alternating) between cities
that are not adjacent in the unconstrained reference tour and do not cross each
other. Concorde solves the TSP with a +100 penalty on every other edge, which
forces the mandatory ones; a tour is kept only if it does not cross itself
(Supp. Sec. C, "Path Constraint Dataset Generation").

Usage (from the repository root, pyconcorde installed):
    python data/make_dataset_path.py --num_cities 20
reads data/tsp20_test_concorde.txt and writes data/tsp20_path_constraint_<yymmdd>.txt.
The released files are the benchmark of record; this script does not reproduce
them bit for bit (see data/README.md).
"""
import argparse
import contextlib
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
from generation_utils import (calculate_distance_matrix, check_tour_intersections,  # noqa: E402
                              sampling_edge, write_tsplib_file)


def main():
    parser = argparse.ArgumentParser(description='Generate the path-constrained TSP benchmark.')
    parser.add_argument('--num_cities', default=20, type=int)
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--work_dir', default='./problem', help='scratch directory for TSPLIB files')
    args = parser.parse_args()

    seed_everything(2024)
    os.makedirs(args.work_dir, exist_ok=True)
    problem_path = os.path.join(args.work_dir, 'tsp_problem.tsp')
    in_path = os.path.join(args.data_path, f'tsp{args.num_cities}_test_concorde.txt')
    out_path = os.path.join(args.data_path,
                            f'tsp{args.num_cities}_path_constraint_{datetime.today().strftime("%y%m%d")}.txt')
    dataset = TSPDataset(data_file=in_path, img_size=64, constraint_type='basic')

    with open(out_path, 'w') as f:
        for i in tqdm(range(len(dataset))):
            points, gt_tour, _ = dataset.parse(i)
            sample_cnt = i % 2 + 1  # 1, 2, 1, 2, ... mandatory edges

            while True:
                edges = sampling_edge(gt_tour, points, sample_cnt=sample_cnt)
                write_tsplib_file(calculate_distance_matrix(points, edges), problem_path)
                with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), \
                        contextlib.redirect_stderr(fnull):
                    solution = TSPSolver.from_tspfile(problem_path).solve()
                route = np.append(solution.tour, solution.tour[0]) + 1
                if not check_tour_intersections(route, points):
                    break

            str_points = ' '.join(str(v) for v in points.flatten().tolist())
            str_tour = ' '.join(str(v) for v in route.flatten().tolist())
            str_path = ' '.join(str(node) for edge in edges for node in edge)
            f.write(f'{str_points} output {str_tour} output {str_path} \n')

            for leftover in glob('*.res') + glob('*.sol'):  # Concorde side files
                os.remove(leftover)


if __name__ == '__main__':
    main()
