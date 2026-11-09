"""Generate the cluster-constrained TSP benchmark from a basic test set.

The cities of instance i are grouped by k-means into k = 3, 4, 5, 6 clusters
(cycling with i). Concorde solves the TSP with a +100 penalty on inter-cluster
edges, so every cluster is visited as one contiguous block
(Supp. Sec. C, "Cluster Constraint Dataset Generation").

Usage (from the repository root, pyconcorde and scikit-learn installed):
    python data/make_dataset_cluster.py --num_cities 20
reads data/tsp20_test_concorde.txt and writes data/tsp20_cluster_constraint_<yymmdd>.txt.
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
from sklearn.cluster import KMeans
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddrl.dataset import TSPDataset  # noqa: E402
from ddrl.utils import seed_everything  # noqa: E402
from generation_utils import (adjust_distances_for_clusters, calculate_distance_matrix,  # noqa: E402
                              write_tsplib_file)


def main():
    parser = argparse.ArgumentParser(description='Generate the cluster-constrained TSP benchmark.')
    parser.add_argument('--num_cities', default=20, type=int)
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--work_dir', default='./problem', help='scratch directory for TSPLIB files')
    args = parser.parse_args()

    seed_everything(2024)
    os.makedirs(args.work_dir, exist_ok=True)
    problem_path = os.path.join(args.work_dir, 'tsp_problem.tsp')
    in_path = os.path.join(args.data_path, f'tsp{args.num_cities}_test_concorde.txt')
    out_path = os.path.join(args.data_path,
                            f'tsp{args.num_cities}_cluster_constraint_{datetime.today().strftime("%y%m%d")}.txt')
    dataset = TSPDataset(data_file=in_path, img_size=64, constraint_type='basic')

    with open(out_path, 'w') as f:
        for i in tqdm(range(len(dataset))):
            points, _, _ = dataset.parse(i)
            k = i % 4 + 3  # 3, 4, 5, 6, 3, ...
            labels = KMeans(n_clusters=k, random_state=0).fit(points).labels_

            distances = adjust_distances_for_clusters(calculate_distance_matrix(points), labels)
            write_tsplib_file(distances, problem_path)
            solution = TSPSolver.from_tspfile(problem_path).solve()
            if solution is None:
                raise RuntimeError(f'Concorde returned no tour for instance {i}')
            route = np.append(solution.tour, solution.tour[0]) + 1

            str_points = ' '.join(str(v) for v in points.flatten().tolist())
            str_tour = ' '.join(str(v) for v in route.flatten().tolist())
            str_label = ' '.join(str(v) for v in labels.flatten().tolist())
            f.write(f'{str_points} output {str_tour} output {str_label} \n')

            for leftover in glob('*.res') + glob('*.sol'):  # Concorde side files
                os.remove(leftover)


if __name__ == '__main__':
    main()
