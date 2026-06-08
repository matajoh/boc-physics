"""Module providing routines for contact point localisation."""

from typing import Tuple

from bocpy import Matrix

from .bodies import Circle, Polygon, RigidBody
from .collisions import Collision


def point_segment_distance(p: Matrix, a: Matrix, b: Matrix) -> float:
    """Find the distance between a point and a line segment."""
    ab = b - a
    ap = p - a
    proj = ap.vecdot(ab)
    ab_length2 = ab.vecdot(ab)
    d = proj / ab_length2
    if d <= 0:
        closest = a
    elif d >= 1:
        closest = b
    else:
        closest = a + ab * d

    return (p - closest).magnitude_squared()


def are_different(a: Matrix, b: Matrix, tol: float) -> bool:
    """Check whether two points differ by more than a tolerance."""
    return abs(a.x - b.x) > tol or abs(a.y - b.y) > tol


def find_contact_points_polygon_polygon(a: Polygon, b: Polygon, tol=1e-5) -> Tuple[Matrix, None]:
    """Find the contact points between two polygons."""
    num_a_vertices = len(a.vertices)
    min_dist = float("inf")
    closest0 = None
    closest1 = None
    for i in range(num_a_vertices):
        v0 = a.transformed_vertices[i]
        v1 = a.transformed_vertices[(i + 1) % num_a_vertices]
        for p in b.transformed_vertices:
            d = point_segment_distance(p, v0, v1)
            if d < min_dist - tol:
                min_dist = d
                closest0 = p
                closest1 = None
            elif d < min_dist + tol and are_different(p, closest0, tol):
                closest1 = p

    num_b_vertices = len(b.vertices)
    for i in range(num_b_vertices):
        v0 = b.transformed_vertices[i]
        v1 = b.transformed_vertices[(i + 1) % num_b_vertices]
        for p in a.transformed_vertices:
            d = point_segment_distance(p, v0, v1)
            if d < min_dist - tol:
                min_dist = d
                closest0 = p
                closest1 = None
            elif d < min_dist + tol and are_different(p, closest0, tol):
                closest1 = p

    return closest0, closest1


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
