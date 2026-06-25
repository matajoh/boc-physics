"""Tests for the Vector2 -> bocpy.Matrix migration (parity and aliasing)."""

import math

from bocpy import Matrix

from bocphysics.bodies import AABB, Circle, Polygon


def test_vecdot_matches_reference_fuzz():
    for _ in range(500):
        a = Matrix.uniform(-10, 10, size=(1, 2))
        b = Matrix.uniform(-10, 10, size=(1, 2))
        ref = a.x * b.x + a.y * b.y
        assert math.isclose(a.vecdot(b), ref, rel_tol=1e-9, abs_tol=1e-12)


def test_cross_matches_reference_fuzz():
    for _ in range(500):
        a = Matrix.uniform(-10, 10, size=(1, 2))
        b = Matrix.uniform(-10, 10, size=(1, 2))
        ref = a.x * b.y - a.y * b.x
        assert math.isclose(a.cross(b), ref, rel_tol=1e-9, abs_tol=1e-12)


def test_perpendicular_is_minus_y_x_fuzz():
    for _ in range(500):
        v = Matrix.uniform(-10, 10, size=(1, 2))
        p = v.perpendicular()
        assert math.isclose(p.x, -v.y)
        assert math.isclose(p.y, v.x)


def test_magnitude_squared_matches_reference_fuzz():
    for _ in range(500):
        v = Matrix.uniform(-10, 10, size=(1, 2))
        ref = v.x * v.x + v.y * v.y
        assert math.isclose(v.magnitude_squared(), ref, rel_tol=1e-9, abs_tol=1e-12)


def test_normalize_unit_length_and_zero_safe():
    v = Matrix.vector([3.0, 4.0]).normalize()
    assert math.isclose(v.length, 1.0)
    z = Matrix.vector([0.0, 0.0]).normalize()
    assert z.x == 0.0 and z.y == 0.0


def test_bodies_have_independent_position_matrices():
    a = Circle.create(1.0, 1.0, (255, 0, 0))
    b = Circle.create(1.0, 1.0, (0, 255, 0))
    assert a.position is not b.position


def test_move_to_copies_the_argument():
    pos = Matrix.vector([5.0, 6.0])
    body = Circle.create(1.0, 1.0, (1, 2, 3)).move_to(pos)
    pos += Matrix.vector([100.0, 100.0])
    assert body.position.x == 5.0 and body.position.y == 6.0


def test_step_integrates_gravity():
    body = Circle.create(1.0, 1.0, (1, 2, 3)).move_to(Matrix.vector([0.0, 0.0]))
    gravity = Matrix.vector([0.0, 10.0])
    body.step(0.1, gravity)
    assert math.isclose(body.linear_velocity.y, 1.0)
    assert math.isclose(body.position.y, 0.1)


def test_aabb_helpers_return_matrices():
    box = AABB(-2.0, -4.0, 6.0, 10.0)
    assert (box.top_left.x, box.top_left.y) == (-2.0, -4.0)
    assert (box.center.x, box.center.y) == (2.0, 3.0)
    assert (box.size.x, box.size.y) == (8.0, 14.0)


def test_rectangle_vertices_are_distinct_matrices():
    rect = Polygon.create_rectangle(2.0, 2.0, 1.0, (1, 2, 3))
    verts = rect.transformed_vertices
    assert len({id(v) for v in verts}) == len(verts)
