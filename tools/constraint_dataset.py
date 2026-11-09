#!/usr/bin/env python3
"""Read, check and render the constrained TSP benchmark files.

Each line of ``data/tsp{N}_{box,path,cluster}_constraint_*.txt`` is::

    x1 y1 x2 y2 ... xN yN output t1 t2 ... tN t1 output <constraint>

``t`` is the Concorde reference tour, 1-based and closed (N+1 entries).
``<constraint>`` depends on the family:

* box (obstacle): four floats ``c0_min c0_max c1_min c1_max`` giving the
  forbidden rectangle in the two coordinate axes; no tour edge may cross it.
* path: ``2k`` integers, 0-based city pairs that must be consecutive in the
  tour; no other tour edge may cross a required edge.
* cluster: ``N`` integers, the cluster id of every city; the tour must enter
  and leave every cluster exactly once.

Usage::

    python tools/constraint_dataset.py check data/tsp20_box_constraint_240710.txt
    python tools/constraint_dataset.py check data/*_constraint_*.txt
    python tools/constraint_dataset.py render data/tsp20_path_constraint_240711.txt 0 out.png

Only numpy is required; ``render`` also needs matplotlib.  The checks follow
the scorer used in the paper (``ddrl/solver.py`` and ``ddrl/geometry.py``), so
the numbers they print are the ones the paper's appendix reports.
"""
import argparse
import os
import sys

import numpy as np

FAMILIES = ('box', 'path', 'cluster')


# ----------------------------------------------------------------------------- parsing
def family_of(path):
    name = os.path.basename(path)
    for fam in FAMILIES:
        if f'_{fam}_constraint_' in name:
            return fam
    raise ValueError(f'cannot tell the constraint family from the file name: {name}')


def parse_line(line, family):
    parts = line.strip().split(' output ')
    if len(parts) != 3:
        raise ValueError('expected two "output" separators')
    pts = np.array([float(t) for t in parts[0].split()]).reshape(-1, 2)
    tour = np.array([int(t) for t in parts[1].split()])
    raw = parts[2].split()
    if family == 'box':
        cons = np.array([float(t) for t in raw])
        if cons.shape != (4,):
            raise ValueError('box needs four numbers')
    elif family == 'path':
        cons = np.array([int(float(t)) for t in raw])
        if len(cons) % 2:
            raise ValueError('path needs an even number of city ids')
    else:
        cons = np.array([int(float(t)) for t in raw])
        if len(cons) != len(pts):
            raise ValueError('cluster needs one id per city')
    return pts, tour, cons


def load(path):
    """Yield (index, (points, tour, constraint)) for every line of a dataset file."""
    fam = family_of(path)
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                yield i, parse_line(line, fam)


def tour_length(points, tour):
    idx = tour - 1 if tour.min() == 1 else tour
    p = points[idx]
    return float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())


# ----------------------------------------------------------------------------- geometry
def _orientation(p, q, r):
    v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if v == 0:
        return 0
    return 1 if v > 0 else 2


def _on_segment(p, q, r):
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))


def segments_intersect(p1, q1, p2, q2):
    """Closed-segment intersection, collinear overlaps included (as in the paper's scorer)."""
    o1, o2 = _orientation(p1, q1, p2), _orientation(p1, q1, q2)
    o3, o4 = _orientation(p2, q2, p1), _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p2, q1):
        return True
    if o2 == 0 and _on_segment(p1, q2, q1):
        return True
    if o3 == 0 and _on_segment(p2, p1, q2):
        return True
    if o4 == 0 and _on_segment(p2, q1, q2):
        return True
    return False


def _strict_cross(p1, p2, q1, q2):
    """Proper crossing only (the test used for required path edges)."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def segment_hits_box(p1, p2, box):
    c0_min, c0_max, c1_min, c1_max = box
    sides = [((c0_min, c1_min), (c0_min, c1_max)), ((c0_min, c1_max), (c0_max, c1_max)),
             ((c0_max, c1_max), (c0_max, c1_min)), ((c0_max, c1_min), (c0_min, c1_min))]
    return any(segments_intersect(p1, p2, a, b) for a, b in sides)


# ----------------------------------------------------------------------------- checks
def check_tour_shape(points, tour):
    n = len(points)
    return len(tour) == n + 1 and tour[0] == tour[-1] and set(tour[:-1]) == set(range(1, n + 1))


def box_violations(points, tour, box):
    """Number of tour edges that touch the forbidden rectangle."""
    t = tour - 1
    return sum(segment_hits_box(points[t[i]], points[t[i + 1]], box) for i in range(len(t) - 1))


def path_violations(points, tour, pairs):
    """(missing adjacencies, crossings, self-loop pairs ignored)."""
    t = list(tour - 1)
    missing = crossing = loops = 0
    for a, b in zip(pairs[0::2], pairs[1::2]):
        if a == b:
            loops += 1          # a degenerate pair; the scorer skips it
            continue
        adjacent = any((t[i] == a and t[i + 1] == b) or (t[i] == b and t[i + 1] == a)
                       for i in range(len(t) - 1))
        if not adjacent:
            missing += 1
        for i in range(len(t) - 1):
            if {a, b} & {t[i], t[i + 1]}:
                continue
            if _strict_cross(points[a], points[b], points[t[i]], points[t[i + 1]]):
                crossing += 1
    return missing, crossing, loops


def cluster_violations(tour, cluster):
    """Number of clusters not entered exactly once and left exactly once."""
    t = list(tour - 1)
    deg = {c: [0, 0] for c in np.unique(cluster)}
    for i in range(len(t) - 1):
        ca, cb = cluster[t[i]], cluster[t[i + 1]]
        if ca != cb:
            deg[ca][1] += 1
            deg[cb][0] += 1
    return sum(1 for c, (i, o) in deg.items() if i != 1 or o != 1)


def check_file(path, verbose=True):
    fam = family_of(path)
    n_lines = bad_shape = 0
    n_viol = 0           # instances whose reference tour violates the constraint
    loops_total = 0
    detail = []
    sizes = set()
    for i, (pts, tour, cons) in load(path):
        n_lines += 1
        sizes.add(len(pts))
        if not check_tour_shape(pts, tour):
            bad_shape += 1
            continue
        if fam == 'box':
            v = box_violations(pts, tour, cons)
            if v:
                n_viol += 1
                detail.append((i, f'{v} edge(s) cross the box'))
        elif fam == 'path':
            missing, crossing, loops = path_violations(pts, tour, cons)
            loops_total += loops
            if missing or crossing:
                n_viol += 1
                detail.append((i, f'{missing} required edge(s) missing, {crossing} crossing(s)'))
        else:
            v = cluster_violations(tour, cons)
            if v:
                n_viol += 1
                detail.append((i, f'{v} cluster(s) not visited contiguously'))
    if verbose:
        print(f'{os.path.basename(path)}: {n_lines} instances, N={sorted(sizes)}, '
              f'malformed tours {bad_shape}, reference tours violating the constraint {n_viol}'
              + (f', self-loop pairs {loops_total}' if fam == 'path' else ''))
        for i, msg in detail[:10]:
            print(f'    instance {i}: {msg}')
        if len(detail) > 10:
            print(f'    ... {len(detail) - 10} more')
    return n_lines, bad_shape, n_viol, loops_total


# ----------------------------------------------------------------------------- rendering
def render(path, index, out, tour_on=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fam = family_of(path)
    target = None
    for i, rec in load(path):
        if i == index:
            target = rec
            break
    if target is None:
        raise SystemExit(f'index {index} not in {path}')
    pts, tour, cons = target
    fig, ax = plt.subplots(figsize=(4, 4))
    if fam == 'box':
        c0_min, c0_max, c1_min, c1_max = cons
        ax.add_patch(plt.Rectangle((c1_min, c0_min), c1_max - c1_min, c0_max - c0_min,
                                   color='0.75', zorder=0))
    if tour_on:
        t = tour - 1
        ax.plot(pts[t, 1], pts[t, 0], '-', color='0.4', lw=1, zorder=1)
    if fam == 'path':
        for a, b in zip(cons[0::2], cons[1::2]):
            ax.plot([pts[a, 1], pts[b, 1]], [pts[a, 0], pts[b, 0]], '-', color='red', lw=2, zorder=2)
    if fam == 'cluster':
        ax.scatter(pts[:, 1], pts[:, 0], c=cons, cmap='tab10', s=18, zorder=3)
    else:
        ax.scatter(pts[:, 1], pts[:, 0], color='black', s=12, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{fam} N={len(pts)} #{index}  len={tour_length(pts, tour):.3f}', fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check', help='verify format and reference-tour feasibility')
    c.add_argument('files', nargs='+')
    r = sub.add_parser('render', help='draw one instance (constraint, reference tour) to PNG')
    r.add_argument('file')
    r.add_argument('index', type=int)
    r.add_argument('out')
    r.add_argument('--no-tour', action='store_true', help='draw cities and constraint only')
    args = ap.parse_args()
    if args.cmd == 'check':
        rc = 0
        for f in args.files:
            n, bad, viol, _ = check_file(f)
            if bad:
                rc = 1
        sys.exit(rc)
    render(args.file, args.index, args.out, tour_on=not args.no_tour)


if __name__ == '__main__':
    main()
