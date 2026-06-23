"""Module providing routines for collision detection."""

import math
from typing import NamedTuple

from bocpy import Matrix

from .bodies import Circle, Polygon, RigidBody


class Collision(NamedTuple("Collision", [("normal", Matrix), ("depth", float)])):
    """A collision described by its contact normal and penetration depth."""

    def reverse(self) -> "Collision":
        """Return a collision with the normal pointing the other way."""
        return Collision(-self.normal, self.depth)


def intersect_circle_circle(a: Circle, b: Circle) -> Collision:
    """Determine if two circles intersect and return the collision."""
    diff = b.position - a.position
    diff_length2 = diff.vecdot(diff)
    # due to the graphics system, a circle is visually a bit smaller
    # than its radius, so this is a bit of a hack to correct for that
    radius_sum = (a.radius + b.radius) * 0.97
    if diff_length2 >= radius_sum**2:
        # this test is equivalent to a test of the actual length but much
        # cheaper to compute by avoiding the square root
        return None

    # this way we only compute the square root if we have to
    diff_length = math.sqrt(diff_length2)
    normal = diff / diff_length
    depth = radius_sum - diff_length
    return Collision(normal, depth)


def closest_vertex_on_polygon(point: Matrix, poly: Polygon) -> Matrix:
    """Find the closest vertex on a polygon to a point."""
    closest = None
    dist = float("inf")
    for v in poly.transformed_vertices:
        d = (point - v).magnitude_squared()
        if d < dist:
            closest = v
            dist = d

    return closest


def intersect_circle_polygon(circle: Circle, poly: Polygon) -> Collision:
    """Determine if a circle and a polygon intersect and return the collision."""
    # the extra axis through the closest vertex covers the circle-at-a-corner case
    closest_point = closest_vertex_on_polygon(circle.position, poly)
    diff = closest_point - circle.position
    normals = Matrix.concat([poly.transformed_normals, diff.normalize()], 0)
    # one batched matmul projects every polygon vertex onto every candidate axis
    poly_proj = poly.transformed_vertices @ normals.T
    rect_min, rect_max = poly_proj.min(axis=0), poly_proj.max(axis=0)
    center = circle.position @ normals.T
    # the circle reads a bit smaller than its radius to match the graphics system
    radius = circle.radius * 0.97
    circ_min, circ_max = center - radius, center + radius
    # any single separating axis rules out the collision outright
    if (Matrix.less(circ_max, rect_min) + Matrix.less(rect_max, circ_min)).max() > 0:
        return None

    # the minimum-overlap axis is the contact normal; argmin keeps the first on ties
    pen1 = circ_max - rect_min
    pen2 = rect_max - circ_min
    depth = Matrix.where(Matrix.less(pen1, pen2), pen1, pen2)
    axis = normals[depth.argmin()]
    if (poly.position - circle.position).vecdot(axis) < 0:
        axis = -axis

    return Collision(axis, depth.min())


def intersect_polygon_polygon(a: Polygon, b: Polygon) -> Collision:
    """Determine if two polygons intersect and return the collision."""
    # one batched matmul projects every vertex of both polygons onto every axis
    normals = Matrix.concat([a.transformed_normals, b.transformed_normals], 0)
    nt = normals.T
    a_proj = a.transformed_vertices @ nt
    b_proj = b.transformed_vertices @ nt
    a_min, a_max = a_proj.min(axis=0), a_proj.max(axis=0)
    b_min, b_max = b_proj.min(axis=0), b_proj.max(axis=0)
    # any single separating axis rules out the collision outright
    if (Matrix.less(a_max, b_min) + Matrix.less(b_max, a_min)).max() > 0:
        return None

    # the minimum-overlap axis is the contact normal; argmin keeps the first on ties
    pen1 = a_max - b_min
    pen2 = b_max - a_min
    depth = Matrix.where(Matrix.less(pen1, pen2), pen1, pen2)
    axis = normals[depth.argmin()]
    if (b.position - a.position).vecdot(axis) < 0:
        axis = -axis

    return Collision(axis, depth.min())


def detect_collision(a: RigidBody, b: RigidBody) -> Collision:
    """Dispatch to the right narrow-phase test for the body pair."""
    if isinstance(a, Circle):
        if isinstance(b, Circle):
            return intersect_circle_circle(a, b)

        return intersect_circle_polygon(a, b)
    elif isinstance(a, Polygon):
        if isinstance(b, Circle):
            collision = intersect_circle_polygon(b, a)
            return collision.reverse() if collision else None

        return intersect_polygon_polygon(a, b)
