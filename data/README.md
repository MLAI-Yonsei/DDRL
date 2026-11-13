# Constrained TSP benchmark: obstacle, path, cluster

Twelve files, one per (constraint family, size), 1,280 instances each. They are
the benchmark used in *DDRL: A Diffusion-Driven Reinforcement Learning Approach
for Enhanced TSP Solutions* (WACV 2027), Tables 2 and 3 and the supplementary material.
`SHA256SUMS` lists the checksum of every file (`sha256sum -c SHA256SUMS`).

| family | files | constraint | sizes |
|---|---|---|---|
| obstacle (`box`) | `tsp{N}_box_constraint_240710.txt` | no tour edge may cross a forbidden rectangle | N = 20, 50, 100, 200 |
| path | `tsp{N}_path_constraint_240711.txt` | given city pairs must be adjacent in the tour, and no other edge may cross those required edges | N = 20, 50, 100, 200 |
| cluster | `tsp{N}_cluster_constraint_240721.txt` | every cluster is entered once and left once (clustered TSP) | N = 20, 50, 100, 200 |

The city coordinates are those of the basic TSP test sets of
[learning-tsp](https://github.com/chaitjo/learning-tsp) (`tsp{N}_test_concorde.txt`,
cities uniform in the unit square), instance for instance, so every constrained
instance has an unconstrained counterpart. The six-digit suffix is the generation
date (2024-07-10/11/21).

## Line format

    x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1 output <constraint>

* `x y`: coordinates in [0, 1]; the first coordinate is the row axis and the
  second the column axis of the rendered image (`utils.draw_tour`).
* `t1 ... tN t1`: the Concorde reference tour, **1-based**, closed (N+1 ids).
* `<constraint>`:
  * obstacle: `c0_min c0_max c1_min c1_max`, the rectangle in the two coordinate
    axes. No city lies inside it. Median area 0.04 at N = 20 and 0.01 at N = 200.
  * path: `2k` **0-based** city ids, `a1 b1 a2 b2 ...`, the k required edges.
    k is 1 to 4 at N = 20 and 50 (320 instances each) and 1 to 2 at N = 100
    and 200 (640 instances each).
  * cluster: N integers in `0..k-1`, the cluster of every city; k is 3 to 6
    (320 instances each, at every size).

`tools/constraint_dataset.py` parses the files, verifies them and renders an
instance:

    python tools/constraint_dataset.py check data/*_constraint_*.txt
    python tools/constraint_dataset.py render data/tsp50_path_constraint_240711.txt 7 inst7.png

## How the reference tours were made

Each family starts from the basic instance and solves the TSP with Concorde
(through pyconcorde) on a modified distance matrix: edges that cross the obstacle
are penalized (penalty 100), required path edges are forced, and the clustered
TSP uses the standard inter-cluster distance transformation. A reference tour is
kept only if it satisfies the constraint under the same scorer the paper uses
(`ddrl/solver.py`). The per-family procedures are written out in the
supplementary material (section *Dataset Generation*).

`make_dataset_{box,path,cluster}.py` are the generation scripts in their latest
revision. They need pyconcorde (and scikit-learn for the cluster script) and are
run from the repository root, e.g. `python data/make_dataset_box.py --num_cities 20`. They do
**not** regenerate these files bit for bit (a later fix to the sampling code
changed, for instance, how many path pairs an instance receives), so treat the
files in this directory as the benchmark of record.

## Known issues (checked with `tools/constraint_dataset.py` on 2026-10-26)

* **Obstacle**: all 4 x 1,280 reference tours avoid the rectangle.
* **Cluster**: all 4 x 1,280 reference tours visit every cluster contiguously.
  At N = 20, 29 of the reference tours are not optimal: feasible tours shorter
  than the reference exist (the paper's solvers find them), so gaps against the
  reference can be slightly negative there. The paper reports this.
* **Path**: some required pairs are degenerate `(a, a)` self-loops, 40 pairs at
  N = 20, 21 at N = 50, 5 at N = 100, none at N = 200. The scorer ignores them;
  they constrain nothing. One reference tour violates its own constraint:
  N = 200, instance 131 omits the required edge (132, 177) and crosses it; its
  length (10.6457) is therefore that of an infeasible tour. The paper's path
  N = 200 numbers include this instance (1 of 1,280).

Please keep the files unchanged if you compare against the paper; fixing the
items above would change the reference lengths.

## License and citation

The coordinates are redistributed from learning-tsp (MIT License). The
constraints and reference tours are released under the MIT License of this
repository. If you use the benchmark, please cite the paper (see the top-level
README for the BibTeX entry).
