"""Tour evaluation, constraint counting and the lexicographic 2-opt (Supp. Sec. E).

For constrained TSP a candidate tour y is compared by the pair
(V(y), l(y)) = (number of violated constraints, tour length): a 2-opt exchange is
accepted if it lowers V, or keeps V and shortens the tour. The keep-best rule of
the search then ranks tours by the penalized cost l(y) + xi * V(y)
(paper, Sec. 3.3; `cost(y)` in Algorithms 1 and 2).

`TwoOptSolver` extends the `TSP_2opt` class of diffusion_priors
(Graikos et al., NeurIPS 2022; https://github.com/AlexGraikos/diffusion_priors,
MIT License, tsp/tsp_utils.py) with constraints, an incremental (delta) move
evaluation, and a repair step for path and cluster constraints.
"""
import os

import numpy as np
import scipy.spatial

from ddrl.geometry import check_consecutive_pair, do_lines_intersect

# Candidate budget of one `_or_opt_general` sweep, which enumerates about
# 6 n (n - 2) moves. The move improves path-constrained tours at small n and is
# too slow to pay off at n = 200, so it is enabled only while one sweep stays
# within this budget (n <= 100). 0 disables it.
_OR_OPT_GENERAL_BUDGET = int(os.environ.get('DDRL_OR_OPT_GENERAL_BUDGET', 60_000))


def count_cluster_violations(cluster, route):
    """Number of clusters that are not entered exactly once and left exactly once."""
    cluster = [int(x) for x in cluster]
    degrees = {c: {'in': 0, 'out': 0} for c in np.unique(cluster)}

    for i in range(len(route) - 1):
        cur, nxt = cluster[route[i]], cluster[route[i + 1]]
        if cur != nxt:
            degrees[cur]['out'] += 1
            degrees[nxt]['in'] += 1
    first, last = cluster[route[0]], cluster[route[-1]]
    if last != first:
        degrees[last]['out'] += 1
        degrees[first]['in'] += 1

    return sum(1 for d in degrees.values() if d['in'] != 1 or d['out'] != 1)


class TwoOptSolver:
    """Length, violation count and lexicographic 2-opt for one instance.

    constraint_type: 'basic' | 'box' | 'path' | 'cluster'
    constraint: box -> (N, N) 0/1 crossing matrix (see geometry.box_crossing_matrix);
                path -> flat list [a1, b1, a2, b2, ...] of mandatory edges;
                cluster -> cluster label per city.
    Routes are closed lists of 0-based city indices ([0, ..., 0]).
    """

    def __init__(self, points, constraint_type, constraint=None):
        self.points = points
        self.dist_mat = scipy.spatial.distance_matrix(points, points)
        self.constraint_type = constraint_type
        if constraint_type == 'box':
            self.constraint_matrix = constraint
        elif constraint_type == 'path':
            self.path_pairs = []
            for i in range(0, len(constraint), 2):
                a, b = int(constraint[i]), int(constraint[i + 1])
                if a == b:
                    # degenerate (a, a) pairs occur in the data; no tour can satisfy
                    # them, so they are ignored (see data/README.md)
                    continue
                self.path_pairs.append((a, b))
            self.path = [n for p in self.path_pairs for n in p]
        elif constraint_type == 'cluster':
            self.cluster = constraint

    # ------------------------------------------------------------------ scoring
    def evaluate(self, route):
        """Tour length l(y)."""
        return sum(self.dist_mat[route[i], route[i + 1]] for i in range(len(route) - 1))

    def count_constraints(self, route):
        """Number of violated constraints V(y).

        box: tour edges that cross the obstacle. path: missing mandatory edges plus
        crossings of a mandatory edge by another tour edge. cluster: clusters not
        visited as one contiguous block.
        """
        count = 0
        if self.constraint_type == 'box':
            for i in range(len(route) - 1):
                if self.constraint_matrix[route[i], route[i + 1]] == 1:
                    count += 1

        if self.constraint_type == 'path':
            for a, b in self.path_pairs:
                if not check_consecutive_pair(route, a, b):
                    count += 1
                pa, pb = self.points[a], self.points[b]
                for j in range(len(route) - 1):
                    if {a, b} & {route[j], route[j + 1]}:
                        continue
                    if do_lines_intersect(pa, pb, self.points[route[j]], self.points[route[j + 1]]):
                        count += 1

        if self.constraint_type == 'cluster':
            count += count_cluster_violations(self.cluster, route)
        return count

    def _score(self, route):
        return self.count_constraints(route), self.evaluate(route)

    # ------------------------------------------------------------------ 2-opt
    def solve_2opt(self, route, max_iter=None):
        """Lexicographic 2-opt. Returns (route, number of sweeps)."""
        assert route[0] == route[-1], 'Tour is not a cycle'
        best = route
        best_constraints_cnt = self.count_constraints(best)
        best_cost = self.evaluate(best)
        improved = True
        steps = 0
        while improved:
            steps += 1
            if max_iter is not None and steps == max_iter:
                break
            improved = False
            base, base_cost, base_cnt = best, best_cost, best_constraints_cnt
            pairs_present = None
            cl_boundary = None
            if self.constraint_type == 'path':
                pairs_present = [check_consecutive_pair(base, a, b) for a, b in self.path_pairs]
            elif self.constraint_type == 'cluster':
                cl_boundary = self._cluster_boundary_counts(base)
            n = len(base)
            for i in range(1, n - 2):
                if self.constraint_type == 'path' and base[i] in self.path:
                    continue  # never break a mandatory edge
                a1, a2 = base[i - 1], base[i]
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    if self.constraint_type == 'path' and base[j] in self.path:
                        continue
                    b1, b2 = base[j - 1], base[j]
                    # reversing base[i:j] replaces edges (a1,a2), (b1,b2) by (a1,b1), (a2,b2)
                    new_cost = base_cost + (self.dist_mat[a1, b1] + self.dist_mat[a2, b2]
                                            - self.dist_mat[a1, a2] - self.dist_mat[b1, b2])
                    new_cnt = base_cnt + self._constraints_delta(
                        ((a1, a2), (b1, b2)), ((a1, b1), (a2, b2)), pairs_present, cl_boundary)

                    if (new_cnt < best_constraints_cnt) or (
                        new_cnt == best_constraints_cnt and new_cost < best_cost
                    ):
                        new_route = base[:]
                        new_route[i:j] = base[j - 1:i - 1:-1]
                        best = new_route
                        # recompute exactly so floating-point drift cannot accumulate
                        best_cost = self.evaluate(new_route)
                        best_constraints_cnt = self.count_constraints(new_route)
                        improved = True
        return best, steps

    def _cluster_boundary_counts(self, route):
        """Inter-cluster tour edges per cluster. On a closed tour a cluster is
        feasible iff this count is exactly 2."""
        cnt = {}
        cl = self.cluster
        for k in range(len(route) - 1):
            cu, cv = int(cl[route[k]]), int(cl[route[k + 1]])
            if cu != cv:
                cnt[cu] = cnt.get(cu, 0) + 1
                cnt[cv] = cnt.get(cv, 0) + 1
        return cnt

    def _constraints_delta(self, removed, added, pairs_present, cl_boundary=None):
        """Change of V(y) caused by a 2-opt move (two edges removed, two added)."""
        if self.constraint_type == 'box':
            return (self.constraint_matrix[added[0]] + self.constraint_matrix[added[1]]
                    - self.constraint_matrix[removed[0]] - self.constraint_matrix[removed[1]])

        if self.constraint_type == 'path':
            delta = 0
            for k, (a, b) in enumerate(self.path_pairs):
                pair = {a, b}
                present_new = pairs_present[k]
                for u, v in removed:
                    if {u, v} == pair:
                        present_new = False
                for u, v in added:
                    if {u, v} == pair:
                        present_new = True
                delta += (not present_new) - (not pairs_present[k])

                pa, pb = self.points[a], self.points[b]
                for u, v in removed:
                    if not (pair & {u, v}) and do_lines_intersect(pa, pb, self.points[u], self.points[v]):
                        delta -= 1
                for u, v in added:
                    if not (pair & {u, v}) and do_lines_intersect(pa, pb, self.points[u], self.points[v]):
                        delta += 1
            return delta

        if self.constraint_type == 'cluster':
            cl = self.cluster
            local = {}
            for (u, v), sgn in ((removed[0], -1), (removed[1], -1), (added[0], 1), (added[1], 1)):
                cu, cv = int(cl[u]), int(cl[v])
                if cu != cv:
                    local[cu] = local.get(cu, 0) + sgn
                    local[cv] = local.get(cv, 0) + sgn
            delta = 0
            for c, d in local.items():
                old = cl_boundary.get(c, 0)
                delta += (old + d != 2) - (old != 2)
            return delta

        return 0  # basic: no constraints

    # ------------------------------------------------------------------ repair
    def repair(self, route, max_rounds=5, kicks=20):
        """Drive an infeasible path- or cluster-constrained tour back to feasibility.

        A 2-opt reversal cannot re-join a split cluster or re-attach a missing
        mandatory edge, so the lexicographic 2-opt can stall at V > 0. This step
        relocates short segments (or-opt) toward feasibility, re-runs the 2-opt,
        and as a last resort applies deterministic double-bridge kicks
        (Supp. Sec. E, "boundary repair"). No-op for other constraint types and
        for feasible tours.
        """
        if self.constraint_type not in ('path', 'cluster'):
            return route
        best = list(route)
        best_v, best_c = self._score(best)
        if best_v == 0:
            return best

        n_open = len(best) - 1
        sweep_cost = sum(n_open * (n_open - L) * 2 for L in (1, 2, 3))
        use_general = self.constraint_type == 'path' and sweep_cost <= _OR_OPT_GENERAL_BUDGET

        for _ in range(max_rounds):
            if best_v == 0:
                break
            if self.constraint_type == 'cluster':
                cand, _ = self._or_opt_cluster_fix(best)
            else:
                cand, _ = self._or_opt_pair_fix(best)
            cand, _ = self.solve_2opt(list(cand))
            v, c = self._score(cand)
            if (v, c) < (best_v, best_c):
                best, best_v, best_c = list(cand), v, c
            if best_v == 0:
                break
            if not use_general:
                break  # the targeted fix has converged
            cand = self._or_opt_general(best)
            v, c = self._score(cand)
            if (v, c) < (best_v, best_c):
                best, best_v, best_c = cand, v, c

        if best_v > 0:
            # deterministic double-bridge kicks
            rng = np.random.default_rng(2024)
            n = len(best) - 1
            for _ in range(kicks):
                open_r = best[:-1]
                i, j, k = sorted(rng.choice(np.arange(1, n), size=3, replace=False))
                cand = open_r[:i] + open_r[j:k] + open_r[i:j] + open_r[k:]
                cand = cand + [cand[0]]
                if self.constraint_type == 'cluster':
                    cand, _ = self._or_opt_cluster_fix(cand)
                else:
                    cand, _ = self._or_opt_pair_fix(cand)
                cand, _ = self.solve_2opt(list(cand))
                if use_general:
                    cand = self._or_opt_general(cand)
                v, c = self._score(cand)
                if (v, c) < (best_v, best_c):
                    best, best_v, best_c = list(cand), v, c
                if best_v == 0:
                    break
        return best

    def _or_opt_pair_fix(self, route):
        """Restore each missing mandatory edge (a, b) by moving one endpoint next to
        the other, or by re-inserting the glued segment a-b anywhere."""
        best = list(route)
        best_v, best_c = self._score(best)
        changed = False
        for a, b in self.path_pairs:
            if check_consecutive_pair(best, a, b):
                continue
            open_r = best[:-1]
            cands = []
            for move_node, anchor in ((b, a), (a, b)):
                rem = [x for x in open_r if x != move_node]
                ai = rem.index(anchor)
                for pos in (ai, ai + 1):
                    cands.append(rem[:pos] + [move_node] + rem[pos:])
            rem2 = [x for x in open_r if x != a and x != b]
            for pos in range(len(rem2) + 1):
                for piece in ([a, b], [b, a]):
                    cands.append(rem2[:pos] + piece + rem2[pos:])
            scored = [(c + [c[0]], self._score(c + [c[0]])) for c in cands
                      if check_consecutive_pair(c + [c[0]], a, b)]
            if not scored:
                continue
            cand, (v, c) = min(scored, key=lambda t: t[1])
            if v <= best_v:
                best, best_v, best_c = cand, v, c
                changed = True
        return best, changed

    def _or_opt_general(self, route, max_sweeps=10):
        """Lexicographic or-opt: relocate segments of length 1 to 3 (first improvement)."""
        best = list(route)
        best_v, best_c = self._score(best)
        for _ in range(max_sweeps):
            improved = False
            open_r = best[:-1]
            n = len(open_r)
            for seg_len in (1, 2, 3):
                for s in range(n):
                    seg = [open_r[(s + k) % n] for k in range(seg_len)]
                    rest = [open_r[(s + seg_len + k) % n] for k in range(n - seg_len)]
                    for pos in range(1, len(rest) + 1):
                        for piece in (seg, seg[::-1]):
                            cand = rest[:pos] + piece + rest[pos:]
                            cand = cand + [cand[0]]
                            v, c = self._score(cand)
                            if v < best_v or (v == best_v and c < best_c - 1e-12):
                                best, best_v, best_c = cand, v, c
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    @staticmethod
    def _cyclic_runs(idx, n):
        """Group sorted positions into maximal cyclically contiguous runs."""
        if not idx:
            return []
        runs, cur = [], [idx[0]]
        for p in idx[1:]:
            if p == cur[-1] + 1:
                cur.append(p)
            else:
                runs.append(cur)
                cur = [p]
        runs.append(cur)
        if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == n - 1:  # wrap-around seam
            runs[0] = runs[-1] + runs[0]
            runs.pop()
        return runs

    def _cluster_excess(self, route):
        """Boundary edges beyond the feasible two per cluster.

        Unlike the violation count, this potential decreases with every merge of a
        stray run and reaches 0 exactly when every cluster is contiguous.
        """
        return sum(abs(k - 2) for k in self._cluster_boundary_counts(route).values())

    def _or_opt_cluster_fix(self, route, max_moves=40):
        """Move the stray runs of each split cluster next to its longest run."""
        cl = self.cluster
        best = list(route)
        best_e, best_c = self._cluster_excess(best), self.evaluate(best)
        changed = False
        for _ in range(max_moves):
            if best_e == 0:
                break
            counts = self._cluster_boundary_counts(best)
            bad = [c for c, k in counts.items() if k != 2]
            if not bad:
                break
            moved = False
            for c in bad:
                open_r = best[:-1]
                n = len(open_r)
                idx = [i for i, node in enumerate(open_r) if int(cl[node]) == c]
                runs = self._cyclic_runs(idx, n)
                if len(runs) < 2:
                    continue
                runs.sort(key=len, reverse=True)
                anchor = runs[0]
                strays = sorted(runs[1:], key=len)  # smallest first
                anchor_ends = (open_r[anchor[0]], open_r[anchor[-1]])
                for stray in strays:
                    drop = set(stray)
                    seg = [open_r[i] for i in stray]
                    rest = [x for j, x in enumerate(open_r) if j not in drop]
                    for end_node in anchor_ends:
                        base_pos = rest.index(end_node)
                        for pos in (base_pos + 1, base_pos):
                            for piece in (seg, seg[::-1]):
                                cand = rest[:pos] + piece + rest[pos:]
                                cand = cand + [cand[0]]
                                e = self._cluster_excess(cand)
                                cst = self.evaluate(cand)
                                if (e, cst) < (best_e, best_c):
                                    best, best_e, best_c = cand, e, cst
                                    moved = changed = True
                                    break
                            if moved:
                                break
                        if moved:
                            break
                    if moved:
                        break
                if moved:
                    break
            if not moved:
                break
        return best, changed
