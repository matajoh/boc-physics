"""Tests for rigid-body factory construction."""

from bocpy import Matrix

from bocphysics.bodies import AABB, Circle, Polygon


def test_static_regular_polygon_fields_not_scrambled():
    polygon = Polygon.create_regular_polygon(5, 2.0, 2.0, (10, 20, 30), is_static=True)
    # a static body has zero inverse mass and inertia
    assert polygon.inv_mass == 0
    assert polygon.inv_inertia == 0
    # the vertices and normals must be the generated lists, not scrambled args
    assert len(polygon.vertices) == 5
    assert all(v.shape == (1, 2) for v in polygon.vertices)
    assert len(polygon.normals) == 5
    assert all(n.shape == (1, 2) for n in polygon.normals)
    assert polygon.color == (10, 20, 30)


def test_dynamic_regular_polygon_has_finite_mass():
    polygon = Polygon.create_regular_polygon(5, 2.0, 2.0, (10, 20, 30))
    assert polygon.inv_mass > 0
    assert polygon.inv_inertia > 0
    assert len(polygon.vertices) == 5
    assert len(polygon.normals) == 5


def test_static_circle_has_no_velocity_attributes():
    circle = Circle.create(1.5, 2.0, (10, 20, 30), is_static=True)
    assert circle.inv_mass == 0
    assert circle.inv_inertia == 0
    # a static body must lack velocity attributes so the engine never integrates it
    assert not hasattr(circle, "linear_velocity")
    assert not hasattr(circle, "angular_velocity")


def test_dynamic_circle_has_velocity_attributes():
    circle = Circle.create(1.5, 2.0, (10, 20, 30))
    assert circle.inv_mass > 0
    assert hasattr(circle, "linear_velocity")
    assert hasattr(circle, "angular_velocity")


def test_sweep_grows_box_along_displacement():
    box = AABB(0, 0, 2, 2)
    swept = box.sweep(Matrix.vector([3, 0]), 0)
    # motion is rightward, so only the right edge extends; left stays put
    assert swept == AABB(0, 0, 5, 2)


def test_sweep_handles_negative_displacement():
    box = AABB(0, 0, 2, 2)
    swept = box.sweep(Matrix.vector([0, -4]), 0)
    # upward motion (negative y) extends the top edge only
    assert swept == AABB(0, -4, 2, 2)


def test_sweep_pads_all_sides_by_slop():
    box = AABB(0, 0, 2, 2)
    swept = box.sweep(Matrix.vector([0, 0]), 0.5)
    # zero displacement, so the box only grows by the slop on every side
    assert swept == AABB(-0.5, -0.5, 2.5, 2.5)


def test_sweep_contains_original_and_moved_box():
    box = AABB(1, 1, 3, 4)
    displacement = Matrix.vector([2, -1])
    swept = box.sweep(displacement, 0.25)
    moved = AABB(box.left + displacement.x, box.top + displacement.y,
                 box.right + displacement.x, box.bottom + displacement.y)
    assert swept.contains(box)
    assert swept.contains(moved)
