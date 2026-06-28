"""Bit-exact parity tests for the batched narrow phase.

The separating-axis test now projects every vertex onto every candidate axis in
one batched matmul instead of looping axis by axis. These tests pin the batched
result against an embedded reference that reproduces the original per-axis loop,
across a fuzz of random poses, so the refactor is proven to match bit-for-bit.
"""

import math
import random

from bocpy import Matrix

from bocphysics.bodies import Circle, Polygon
from bocphysics.collisions import (batched_circle_circle, batched_circle_polygon,
                                   closest_vertex_on_polygon, Collision,
                                   detect_collision, intersect_circle_circle,
                                   intersect_circle_polygon)

COLOR = (180, 90, 90)


def ref_intersect_polygon_polygon(a, b):
    """Original per-axis separating-axis test, kept as the parity oracle."""
    normals = Matrix.concat([a.transformed_normals, b.transformed_normals], 0)
    axis = None
    min_depth = float("inf")
    for normal in normals:
        a_proj = a.transformed_vertices.vecdot(normal, axis=1)
        b_proj = b.transformed_vertices.vecdot(normal, axis=1)
        a_min, a_max = a_proj.min(), a_proj.max()
        b_min, b_max = b_proj.min(), b_proj.max()
        if a_max < b_min or b_max < a_min:
            return None

        depth = min(a_max - b_min, b_max - a_min)
        if depth < min_depth:
            min_depth = depth
            axis = normal

    if (b.position - a.position).vecdot(axis) < 0:
        axis = -axis

    return Collision(axis, min_depth)


def ref_intersect_circle_polygon(circle, poly):
    """Original per-axis circle-polygon test, kept as the parity oracle."""
    closest_point = closest_vertex_on_polygon(circle.position, poly)
    diff = closest_point - circle.position
    normals = Matrix.concat([poly.transformed_normals, diff.normalize()], 0)
    axis = None
    min_depth = float("inf")
    radius = circle.radius * 0.97
    for normal in normals:
        center = circle.position.vecdot(normal)
        c_min, c_max = center - radius, center + radius
        rect = poly.transformed_vertices.vecdot(normal, axis=1)
        r_min, r_max = rect.min(), rect.max()
        if c_max < r_min or r_max < c_min:
            return None

        depth = min(c_max - r_min, r_max - c_min)
        if depth < min_depth:
            min_depth = depth
            axis = normal

    if (poly.position - circle.position).vecdot(axis) < 0:
        axis = -axis

    return Collision(axis, min_depth)


def ref_detect_collision(a, b):
    """Dispatch a body pair to the reference per-axis narrow phase."""
    if isinstance(a, Circle):
        if isinstance(b, Circle):
            return intersect_circle_circle(a, b)

        return ref_intersect_circle_polygon(a, b)

    if isinstance(b, Circle):
        collision = ref_intersect_circle_polygon(b, a)
        return collision.reverse() if collision else None

    return ref_intersect_polygon_polygon(a, b)


def make_body(rng):
    """Build a randomly shaped, randomly posed rectangle, polygon, or circle."""
    kind = rng.choice(["rect", "rect", "poly", "circle"])
    if kind == "circle":
        body = Circle.create(rng.uniform(0.4, 1.3), 1.0, COLOR)
    elif kind == "poly":
        body = Polygon.create_regular_polygon(rng.choice([3, 5, 6]),
                                              rng.uniform(0.5, 1.3), 1.0, COLOR)
    else:
        body = Polygon.create_rectangle(rng.uniform(0.6, 2.0),
                                        rng.uniform(0.6, 2.0), 1.0, COLOR)

    return body


def place(body, x, y, angle):
    """Move a freshly built body to an absolute pose."""
    body.move_to(Matrix.vector([x, y]))
    body.angle = angle
    body.update_needed_ = True
    return body


def collisions_equal(p, q):
    """Bit-exact equality on two optional Collision results."""
    if p is None or q is None:
        return p is None and q is None

    return p.normal.x == q.normal.x and p.normal.y == q.normal.y and p.depth == q.depth


def ref_closest_vertex_on_polygon(point, poly):
    """Original per-vertex closest-vertex loop, kept as the parity oracle."""
    closest = None
    dist = float("inf")
    for v in poly.transformed_vertices:
        d = (point - v).magnitude_squared()
        if d < dist:
            closest = v
            dist = d

    return closest


def test_closest_vertex_matches_reference():
    """The argmin closest-vertex search equals the per-vertex loop bit-for-bit."""
    rng = random.Random(20260628)
    for _ in range(2000):
        poly = place(make_body(rng), 0.0, 0.0, rng.uniform(-math.pi, math.pi))
        while isinstance(poly, Circle):
            poly = place(make_body(rng), 0.0, 0.0, rng.uniform(-math.pi, math.pi))
        point = Matrix.vector([rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)])
        reference = ref_closest_vertex_on_polygon(point, poly)
        batched = closest_vertex_on_polygon(point, poly)
        assert batched.x == reference.x and batched.y == reference.y


def test_batched_sat_matches_reference():
    """The batched narrow phase equals the per-axis reference bit-for-bit."""
    rng = random.Random(20260619)
    hits = 0
    for k in range(2000):
        a = place(make_body(rng), 0.0, 0.0, rng.uniform(-math.pi, math.pi))
        b = place(make_body(rng), rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6),
                  rng.uniform(-math.pi, math.pi))
        a.uid, b.uid = 2 * k, 2 * k + 1
        for left, right in ((a, b), (b, a)):
            reference = ref_detect_collision(left, right)
            batched = detect_collision(left, right)
            if reference is not None:
                hits += 1

            assert collisions_equal(reference, batched)

    assert hits > 0


def make_circle(rng):
    """Build a randomly sized, randomly placed circle."""
    body = Circle.create(rng.uniform(0.4, 1.3), 1.0, COLOR)
    return place(body, rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6), 0.0)


def make_polygon(rng):
    """Build a randomly shaped, randomly posed polygon (never a circle)."""
    body = make_body(rng)
    while isinstance(body, Circle):
        body = make_body(rng)
    return place(body, rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6),
                 rng.uniform(-math.pi, math.pi))


def test_batched_circle_circle_matches_reference():
    """The batched circle-circle test equals the per-pair oracle bit-for-bit."""
    rng = random.Random(20260628)
    pairs = []
    for _ in range(2000):
        a = make_circle(rng)
        b = make_circle(rng)
        pairs.append((a, b))
    batched = batched_circle_circle(pairs)
    hits = 0
    for (a, b), got in zip(pairs, batched):
        ref = intersect_circle_circle(a, b)
        if ref is not None:
            hits += 1
        assert collisions_equal(ref, got)
    assert hits > 0


def test_batched_circle_polygon_matches_reference():
    """The batched circle-polygon test equals the per-pair oracle bit-for-bit."""
    rng = random.Random(20260629)
    pairs = []
    for _ in range(2000):
        pairs.append((make_circle(rng), make_polygon(rng)))
    batched = batched_circle_polygon(pairs)
    hits = 0
    for (c, p), got in zip(pairs, batched):
        ref = intersect_circle_polygon(c, p)
        if ref is not None:
            hits += 1
        assert collisions_equal(ref, got)
    assert hits > 0
