"""Module providing routines for contact point localisation."""

from typing import Tuple

from bocpy import Matrix

from .bodies import Circle, Polygon, RigidBody
from .collisions import Collision


def points_segment_distances(points: Matrix, a: Matrix, b: Matrix) -> list:
    """Find the squared distances from every point in a block to one segment.

    Description:
        The projection parameter, the clamp onto the segment, and the residual
        are all computed once over the whole (N x 2) point block rather than per
        point. Clipping the parameter to [0, 1] clamps the foot of the
        projection to the segment's endpoints.
    """
    ab = b - a
    t = (points - a).vecdot(ab, axis=1) / ab.vecdot(ab)
    # t @ ab is the outer product scaling ab per row, so closest is an (N x 2) block
    closest = a + t.clip(0, 1) @ ab
    return list((points - closest).magnitude_squared(axis=1))


def scan_edge_points(points: Matrix, distances: list, tol: float, state: list):
    """Fold one edge's point distances into the running closest-point manifold.

    Description:
        state is [min_dist, closest0, closest1]. Points within tol of the
        running minimum form a two-point manifold; a strictly closer point
        resets it. This mirrors the original sequential scan exactly.
    """
    for i, d in enumerate(distances):
        if d < state[0] - tol:
            state[0] = d
            state[1] = points[i]
            state[2] = None
        elif d < state[0] + tol and are_different(points[i], state[1], tol):
            state[2] = points[i]


def are_different(a: Matrix, b: Matrix, tol: float) -> bool:
    """Check whether two points differ by more than a tolerance."""
    return abs(a.x - b.x) > tol or abs(a.y - b.y) > tol


def find_contact_points_polygon_polygon(a: Polygon, b: Polygon, tol=1e-5) -> Tuple[Matrix, None]:
    """Find the contact points between two polygons."""
    a_verts = a.transformed_vertices
    b_verts = b.transformed_vertices
    # state is [min_dist, closest0, closest1] shared across both edge sweeps
    state = [float("inf"), None, None]
    scan_polygon_edges(a_verts, b_verts, tol, state)
    scan_polygon_edges(b_verts, a_verts, tol, state)
    return state[1], state[2]


def scan_polygon_edges(edges: Matrix, points: Matrix, tol: float, state: list):
    """Sweep every edge of one polygon against all of another's vertices."""
    num_edges = edges.rows
    for i in range(num_edges):
        v0 = edges[i]
        v1 = edges[(i + 1) % num_edges]
        distances = points_segment_distances(points, v0, v1)
        scan_edge_points(points, distances, tol, state)


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
                        collision: Collision) -> Tuple[Matrix, Matrix]:
    """Find the contact points for a collision, then push the bodies apart.

    Description:
        Contact points are computed from the overlapping configuration, before
        any position correction, so the manifold reflects the true contact.
        Only then are the bodies separated along the collision normal.
    """
    if isinstance(a, Circle):
        points = a.position + collision.normal * a.radius, None
    elif isinstance(b, Circle):
        points = b.position - collision.normal * b.radius, None
    else:
        points = find_contact_points_polygon_polygon(a, b)

    separate(a, b, collision)
    return points
