"""Module providing routines for contact point localisation."""

from typing import Tuple

from bocpy import Matrix

from . import transport
from .bodies import Circle, Polygon, RigidBody
from .collisions import Collision

_BIG = 1e30


def edge_point_distances(edges: Matrix, points: Matrix) -> Matrix:
    """Find the squared distance from every point to every edge as an (E x P) block.

    Description:
        One batched pass replaces the per-edge projection loop. The point-to-
        segment squared distance is expanded algebraically so the whole
        (edges x points) grid falls out of 2D matrix products: project every
        point onto every edge direction, clamp the foot to the segment, and read
        the residual. The (E x P) matmul term leads each broadcast so the point
        row vector and the edge column vector fold in without a temporary.
    """
    rows = edges.rows
    v0 = edges
    v1 = edges[[(i + 1) % rows for i in range(rows)]]
    ab = v1 - v0
    length2 = ab.magnitude_squared(axis=1)
    pt = points.T
    proj = ab @ pt - v0.vecdot(ab, axis=1)
    t = (proj / length2).clip(0, 1)
    q2 = (v0 @ pt) * -2 + points.magnitude_squared(axis=1).T + v0.magnitude_squared(axis=1)
    return q2 - t * proj * 2 + t * t * length2


def _closest_edge_dist(ex: Matrix, ey: Matrix, px: Matrix, py: Matrix,
                       vmax: int, nedges: Matrix) -> Matrix:
    """Per-point closest-edge squared distance for many pairs, masked to real edges.

    Description:
        Reduces the edge axis of batched_edge_point_blocks down to one distance
        per point: a vertex-major running minimum over edges. nedges is the per-
        pair edge count (N x 1); edge e folds into a pair's minimum only while
        e < nedges, so padded and degenerate edges never win.
    """
    n = ex.rows
    dist = Matrix.full((n, vmax), _BIG)
    for e in range(vmax):
        nxt = (e + 1) % vmax
        v0x, v0y = ex[:, e], ey[:, e]
        abx = ex[:, nxt] - v0x
        aby = ey[:, nxt] - v0y
        length2 = abx * abx + aby * aby
        proj = abx * px + aby * py - (v0x * abx + v0y * aby)
        t = (proj / length2).clip(0, 1)
        q2 = (v0x * px + v0y * py) * -2 + (px * px + py * py) + (v0x * v0x + v0y * v0y)
        block = q2 - t * proj * 2 + t * t * length2
        cond = Matrix.less(block, dist) * Matrix.greater(nedges, float(e))
        dist = Matrix.where(cond, block, dist)
    return dist


def _point_mask(n: int, vmax: int, nvb: list, nva: list) -> Matrix:
    """Column-validity mask over each pair's [B-points | A-points] concatenated row."""
    mask = Matrix.zeros((n, 2 * vmax))
    for k in range(n):
        mask[k, :nvb[k]] = 1.0
        mask[k, vmax:vmax + nva[k]] = 1.0
    return mask


def batched_contact_points(geom, pairs: list, tol=1e-5, state=None) -> Matrix:
    """Localise the two-point contact manifold for every polygon pair at once.

    Description:
        Relaxed order-independent semantics with a stable canonical tie-break.
        The contact set is every vertex within tol of the pair's minimum vertex-
        to-edge distance. contact0 is that set's lowest column index -- a fixed
        per-contact choice that does not flip between within-tol-equidistant
        points as the pose wiggles, unlike a raw global argmin, which would jerk
        the contact lever arm and inject energy into the stiff colour solver.
        contact1 is the set member spatially farthest from contact0, present only
        when it clears tol; farthest (not nearest) spans the contact face, since
        each corner carries two near-coincident vertex attributions a nearest
        rule could collapse onto one side. Returns an (n x 13) Matrix aligned with
        pairs; column 0 is count (1 or 2) and each contact point occupies the next
        six columns [px, py, ra_x, ra_y, rb_x, rb_y] -- its world position and the
        lever arms point - centre_a and point - centre_b, where each centre is
        the body's State-block pose when state is given (else its scalar pose).
        The second point's block is unused when count is 1. Both scan directions are collapsed to
        per-point closest-edge distances, concatenated into one row per pair; a
        masked min picks the contact band, argmax over the validity mask picks
        contact0, and a within-band argmax over Chebyshev separation picks
        contact1 -- no per-cell Python scan.
    """
    n = len(pairs)
    vmax = geom.vmax

    rows_a = [None] * n
    rows_b = [None] * n
    pos_a = [None] * n
    pos_b = [None] * n
    for i, (a, b) in enumerate(pairs):
        rows_a[i] = geom.row_of[a.uid]
        pos_a[i] = transport.block_center(a, state)
        rows_b[i] = geom.row_of[b.uid]
        pos_b[i] = transport.block_center(b, state)

    ax, ay = geom.geom_x.take(rows_a, 0), geom.geom_y.take(rows_a, 0)
    bx, by = geom.geom_x.take(rows_b, 0), geom.geom_y.take(rows_b, 0)
    nva = [len(a.vertices) for a, _ in pairs]
    nvb = [len(b.vertices) for _, b in pairs]
    na = Matrix(n, 1, nva)
    nb = Matrix(n, 1, nvb)
    ab = _closest_edge_dist(ax, ay, bx, by, vmax, na)
    ba = _closest_edge_dist(bx, by, ax, ay, vmax, nb)
    dist = Matrix.concat([ab, ba], 1)
    colx = Matrix.concat([bx, ax], 1)
    coly = Matrix.concat([by, ay], 1)
    pmask = _point_mask(n, vmax, nvb, nva)

    d_min = dist.min(axis=1)
    within = dist.less_equal(d_min + tol) * pmask

    c0m = within.argmax(axis=1)
    x0m = colx.take_along_axis(c0m, axis=1)
    y0m = coly.take_along_axis(c0m, axis=1)

    dx = (colx - x0m).abs()
    dy = (coly - y0m).abs()
    sep = Matrix.where(dx.greater(dy), dx, dy)

    c1m = sep.argmax(axis=1, where=within)
    x1m = colx.take_along_axis(c1m, axis=1)
    y1m = coly.take_along_axis(c1m, axis=1)
    sepmax = sep.take_along_axis(c1m, axis=1)

    p0 = Matrix.concat([x0m, y0m], axis=1)
    p1 = Matrix.concat([x1m, y1m], axis=1)
    ca = Matrix.concat(pos_a)
    cb = Matrix.concat(pos_b)
    result = Matrix.zeros((n, 13))
    result[:, 0:1] = sepmax.greater(tol) + 1.0
    result[:, 1:3] = p0
    result[:, 3:5] = p0 - ca
    result[:, 5:7] = p0 - cb
    result[:, 7:9] = p1
    result[:, 9:11] = p1 - ca
    result[:, 11:13] = p1 - cb
    return result


def scan_edge_points(points: Matrix, distances: list, tol: float, state: list,
                     source_uid):
    """Fold one edge's point distances into the running closest-point manifold.

    Description:
        state is [min_dist, closest0, closest1, id0, id1]. Points within tol of
        the running minimum form a two-point manifold; a strictly closer point
        resets it. The point assignment mirrors the original sequential scan
        exactly; the feature IDs (source_uid, vertex_index) ride alongside and
        never influence which point is chosen.
    """
    for i, d in enumerate(distances):
        if d < state[0] - tol:
            state[0] = d
            state[1] = points[i]
            state[2] = None
            state[3] = (source_uid, i)
            state[4] = None
        elif d < state[0] + tol and are_different(points[i], state[1], tol):
            state[2] = points[i]
            state[4] = (source_uid, i)


def are_different(a: Matrix, b: Matrix, tol: float) -> bool:
    """Check whether two points differ by more than a tolerance."""
    return abs(a.x - b.x) > tol or abs(a.y - b.y) > tol


def find_contact_points_polygon_polygon(a: Polygon, b: Polygon, geom, tol=1e-5) -> Tuple:
    """Find the contact points between two polygons, with feature IDs.

    Description:
        Returns (closest0, closest1, id0, id1). Each contact point is a vertex
        of one of the polygons, so its feature ID is (source_uid, vertex_index)
        -- the natural stable identity for warm-starting. The IDs never affect
        the points, which stay byte-identical to the pre-ID scan. Vertices come
        from the shared GeometryPool, bit-identical to transformed_vertices.
    """
    a_verts = geom.world_vertices(a.uid)
    b_verts = geom.world_vertices(b.uid)
    state = [float("inf"), None, None, None, None]
    scan_polygon_edges(a_verts, b_verts, tol, state, b.uid)
    scan_polygon_edges(b_verts, a_verts, tol, state, a.uid)
    return state[1], state[2], state[3], state[4]


def scan_polygon_edges(edges: Matrix, points: Matrix, tol: float, state: list,
                       source_uid):
    """Sweep every edge of one polygon against all of another's vertices."""
    distances = edge_point_distances(edges, points)
    num_points = points.rows
    for i in range(edges.rows):
        row = [distances[i, j] for j in range(num_points)]
        scan_edge_points(points, row, tol, state, source_uid)


def find_contact_points(a: RigidBody,
                        b: RigidBody,
                        collision: Collision,
                        geom,
                        state=None) -> Tuple:
    """Find the contact points for a collision from the overlapping configuration.

    Description:
        A pure geometry query with no side effects, returning (c0, c1, id0, id1).
        The contact points are read from the current overlapping poses; each
        carries a (source_uid, vertex_index) feature ID for warm-starting. A
        circle contributes its single surface point with ID (circle_uid, 0). When
        state is given, a circle's centre is sourced from the State block.
        Positional correction is the caller's responsibility -- call separate
        explicitly afterwards.
    """
    if isinstance(a, Circle):
        points = transport.block_center(a, state) + collision.normal * a.radius, None, (a.uid, 0), None
    elif isinstance(b, Circle):
        points = transport.block_center(b, state) - collision.normal * b.radius, None, (b.uid, 0), None
    else:
        points = find_contact_points_polygon_polygon(a, b, geom)

    return points
