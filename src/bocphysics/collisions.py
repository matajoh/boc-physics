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
    # shrink the hitbox ~3% so drawn circles appear to touch at contact
    radius_sum = (a.radius + b.radius) * 0.97
    if diff_length2 >= radius_sum**2:
        return None

    diff_length = math.sqrt(diff_length2)
    normal = diff / diff_length
    depth = radius_sum - diff_length
    return Collision(normal, depth)


def closest_vertex_on_polygon(point: Matrix, poly: Polygon) -> Matrix:
    """Find the closest vertex on a polygon to a point."""
    verts = poly.transformed_vertices
    return verts[(verts - point).magnitude_squared(axis=1).argmin()]


def intersect_circle_polygon(circle: Circle, poly: Polygon) -> Collision:
    """Determine if a circle and a polygon intersect and return the collision."""
    closest_point = closest_vertex_on_polygon(circle.position, poly)
    diff = closest_point - circle.position
    normals = Matrix.concat([poly.transformed_normals, diff.normalize()], 0)
    poly_proj = poly.transformed_vertices @ normals.T
    rect_min, rect_max = poly_proj.min(axis=0), poly_proj.max(axis=0)
    center = circle.position @ normals.T
    # shrink the hitbox ~3% so drawn circles appear to touch at contact
    radius = circle.radius * 0.97
    circ_min, circ_max = center - radius, center + radius
    if (Matrix.less(circ_max, rect_min) + Matrix.less(rect_max, circ_min)).max() > 0:
        return None

    pen1 = circ_max - rect_min
    pen2 = rect_max - circ_min
    depth = Matrix.where(Matrix.less(pen1, pen2), pen1, pen2)
    axis = normals[depth.argmin()]
    if (poly.position - circle.position).vecdot(axis) < 0:
        axis = -axis

    return Collision(axis, depth.min())


def intersect_polygon_polygon(a: Polygon, b: Polygon) -> Collision:
    """Determine if two polygons intersect and return the collision."""
    normals = Matrix.concat([a.transformed_normals, b.transformed_normals], 0)
    nt = normals.T
    a_proj = a.transformed_vertices @ nt
    b_proj = b.transformed_vertices @ nt
    a_min, a_max = a_proj.min(axis=0), a_proj.max(axis=0)
    b_min, b_max = b_proj.min(axis=0), b_proj.max(axis=0)
    if (Matrix.less(a_max, b_min) + Matrix.less(b_max, a_min)).max() > 0:
        return None

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
