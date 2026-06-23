"""Module providing routines for contact point localisation."""

from typing import Tuple

from bocpy import Matrix

from .bodies import Circle, Polygon, RigidBody
from .collisions import Collision


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
    # proj is (p - v0) . ab for every (edge, point) pair; the clamp gives the foot
    proj = ab @ pt - v0.vecdot(ab, axis=1)
    t = (proj / length2).clip(0, 1)
    # |p - v0|^2 expanded so the closest-point residual stays a pure 2D expression
    q2 = (v0 @ pt) * -2 + points.magnitude_squared(axis=1).T + v0.magnitude_squared(axis=1)
    return q2 - t * proj * 2 + t * t * length2


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


def find_contact_points_polygon_polygon(a: Polygon, b: Polygon, tol=1e-5) -> Tuple:
    """Find the contact points between two polygons, with feature IDs.

    Description:
        Returns (closest0, closest1, id0, id1). Each contact point is a vertex
        of one of the polygons, so its feature ID is (source_uid, vertex_index)
        -- the natural stable identity for warm-starting. The IDs never affect
        the points, which stay byte-identical to the pre-ID scan.
    """
    a_verts = a.transformed_vertices
    b_verts = b.transformed_vertices
    # state is [min_dist, closest0, closest1, id0, id1] shared across both sweeps
    state = [float("inf"), None, None, None, None]
    # contact points are vertices of the swept (points) polygon, so b then a
    scan_polygon_edges(a_verts, b_verts, tol, state, b.uid)
    scan_polygon_edges(b_verts, a_verts, tol, state, a.uid)
    return state[1], state[2], state[3], state[4]


def scan_polygon_edges(edges: Matrix, points: Matrix, tol: float, state: list,
                       source_uid):
    """Sweep every edge of one polygon against all of another's vertices."""
    distances = edge_point_distances(edges, points)
    num_points = points.rows
    for i in range(edges.rows):
        # replay the running-min selection per edge over the pre-built distances
        row = [distances[i, j] for j in range(num_points)]
        scan_edge_points(points, row, tol, state, source_uid)


def separate(a: RigidBody, b: RigidBody, collision: Collision):
    """Push the bodies apart along the collision normal to remove the overlap."""
    correction = collision.depth
    if not a.physics:
        # b does not move
        b.move(collision.normal * correction)
    elif not b.physics:
        # a does not move
        a.move(-collision.normal * correction)
    else:
        a.move(-collision.normal * correction * 0.5)
        b.move(collision.normal * correction * 0.5)


def find_contact_points(a: RigidBody,
                        b: RigidBody,
                        collision: Collision) -> Tuple:
    """Find the contact points for a collision from the overlapping configuration.

    Description:
        A pure geometry query with no side effects, returning (c0, c1, id0, id1).
        The contact points are read from the current overlapping poses; each
        carries a (source_uid, vertex_index) feature ID for warm-starting. A
        circle contributes its single surface point with ID (circle_uid, 0).
        Positional correction is the caller's responsibility -- call separate
        explicitly afterwards.
    """
    if isinstance(a, Circle):
        points = a.position + collision.normal * a.radius, None, (a.uid, 0), None
    elif isinstance(b, Circle):
        points = b.position - collision.normal * b.radius, None, (b.uid, 0), None
    else:
        points = find_contact_points_polygon_polygon(a, b)

    return points
