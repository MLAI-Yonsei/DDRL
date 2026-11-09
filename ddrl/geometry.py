"""Planar geometry used by the constraint checks: segment intersection and boxes.

All tests operate on city coordinates in the unit square. `do_intersect` is the
reference predicate (orientation test with the collinear special cases);
`segments_intersect_any` is its vectorized, bit-exact counterpart used on the hot
path of the constrained decoder.
"""
import numpy as np


def check_consecutive_pair(route, a, b):
    """True if cities a and b are adjacent somewhere in `route` (either order)."""
    for i in range(len(route) - 1):
        if (route[i] == a and route[i + 1] == b) or (route[i] == b and route[i + 1] == a):
            return True
    return False


def do_lines_intersect(p1, p2, q1, q2):
    """Proper-crossing test of segments p1p2 and q1q2 (counter-clockwise rule).

    Used to count violations of the path constraint: a mandatory edge must not be
    crossed by any other tour edge.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


def orientation(p, q, r):
    """Orientation of the ordered triplet (p, q, r): 0 collinear, 1 clockwise, 2 counter-clockwise."""
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2


def on_segment(p, q, r):
    """True if q lies inside the bounding box of segment pr (used after a collinearity test)."""
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))


def do_intersect(p1, q1, p2, q2):
    """True if segments p1q1 and p2q2 intersect, touching endpoints included."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def segments_intersect_any(p1, q1, P2, Q2):
    """Vectorized, bit-exact equivalent of
    ``any(do_intersect(p1, q1, P2[k], Q2[k]) for k in range(len(P2)))``.

    p1, q1 : (2,) endpoints of the candidate segment
    P2, Q2 : (m, 2) endpoints of the existing segments

    The orientation values are computed with the same IEEE-754 operations as the
    scalar version, so boundary cases (shared endpoints, collinear segments) are
    decided identically.
    """
    if len(P2) == 0:
        return False
    p1x, p1y = p1[0], p1[1]
    q1x, q1y = q1[0], q1[1]
    P2x, P2y = P2[:, 0], P2[:, 1]
    Q2x, Q2y = Q2[:, 0], Q2[:, 1]

    # orientation(p, q, r) = (q1 - p1) * (r0 - q0) - (q0 - p0) * (r1 - q1)
    d1y = q1y - p1y
    d1x = q1x - p1x
    v1 = d1y * (P2x - q1x) - d1x * (P2y - q1y)   # orientation(p1, q1, P2)
    v2 = d1y * (Q2x - q1x) - d1x * (Q2y - q1y)   # orientation(p1, q1, Q2)
    d2y = Q2y - P2y
    d2x = Q2x - P2x
    v3 = d2y * (p1x - Q2x) - d2x * (p1y - Q2y)   # orientation(P2, Q2, p1)
    v4 = d2y * (q1x - Q2x) - d2x * (q1y - Q2y)   # orientation(P2, Q2, q1)

    # Comparing orientation codes is the same as comparing signs
    # (0 <-> 0, 1 <-> +1, 2 <-> -1 is a bijection).
    s1, s2, s3, s4 = np.sign(v1), np.sign(v2), np.sign(v3), np.sign(v4)

    # general position: o1 != o2 and o3 != o4
    if bool(((s1 != s2) & (s3 != s4)).any()):
        return True

    # collinear special cases: only rows with a zero orientation can still match
    z = (s1 == 0) | (s2 == 0) | (s3 == 0) | (s4 == 0)
    if not bool(z.any()):
        return False
    idx = np.nonzero(z)[0]
    P2x, P2y, Q2x, Q2y = P2x[idx], P2y[idx], Q2x[idx], Q2y[idx]
    s1, s2, s3, s4 = s1[idx], s2[idx], s3[idx], s4[idx]

    lo1x, hi1x = min(p1x, q1x), max(p1x, q1x)
    lo1y, hi1y = min(p1y, q1y), max(p1y, q1y)
    res = (s1 == 0) & (P2x <= hi1x) & (P2x >= lo1x) & (P2y <= hi1y) & (P2y >= lo1y)
    res |= (s2 == 0) & (Q2x <= hi1x) & (Q2x >= lo1x) & (Q2y <= hi1y) & (Q2y >= lo1y)

    lo2x, hi2x = np.minimum(P2x, Q2x), np.maximum(P2x, Q2x)
    lo2y, hi2y = np.minimum(P2y, Q2y), np.maximum(P2y, Q2y)
    res |= (s3 == 0) & (p1x <= hi2x) & (p1x >= lo2x) & (p1y <= hi2y) & (p1y >= lo2y)
    res |= (s4 == 0) & (q1x <= hi2x) & (q1x >= lo2x) & (q1y <= hi2y) & (q1y >= lo2y)
    return bool(res.any())


def does_intersect_box(p1, p2, box):
    """True if segment p1p2 crosses one of the four sides of `box` = (x_left, x_right, y_bottom, y_top)."""
    x_left, x_right, y_bottom, y_top = box
    edges = [
        ((x_left, y_bottom), (x_left, y_top)),
        ((x_left, y_top), (x_right, y_top)),
        ((x_right, y_top), (x_right, y_bottom)),
        ((x_right, y_bottom), (x_left, y_bottom)),
    ]
    return any(do_intersect(p1, p2, p3, q3) for p3, q3 in edges)


def box_crossing_matrix(points, box):
    """Obstacle constraint as an (N, N) 0/1 matrix: entry 1 iff edge (i, j) crosses the box.

    The decoder skips such edges and the 2-opt counts them as violations.
    """
    num_points = points.shape[0]
    crossing = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if does_intersect_box(points[i], points[j], box):
                crossing[i, j] = crossing[j, i] = 1
    return crossing
