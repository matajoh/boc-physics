"""Tests for the position-correction step in the contact solver."""

from bocpy import Matrix

from bocphysics.bodies import Polygon
from bocphysics.collisions import Collision, detect_collision
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.contacts import find_contact_points, separate
from bocphysics.engine import PhysicsEngine

UP = Matrix.vector([0, 1])


def make_engine() -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                         DetectionKind.QUADTREE, show_contacts=False)


def test_separate_removes_the_full_penetration():
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    floor.move_to(Matrix.vector([0, 10]))
    box.move_to(Matrix.vector([0, 8.5]))
    floor.physics, box.physics = False, True
    before = box.position.y
    depth = 0.5

    separate(box, floor, Collision(UP, depth))

    # the static floor stays put, so the dynamic box absorbs the whole overlap
    assert abs((before - box.position.y) - depth) < 1e-9


def test_separate_splits_correction_between_two_dynamic_bodies():
    lower = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    upper = Polygon.create_rectangle(2, 2, 2.0, (50, 100, 200))
    lower.move_to(Matrix.vector([0, 10]))
    upper.move_to(Matrix.vector([0, 8.5]))
    lower.physics, upper.physics = True, True
    depth = 0.5

    separate(upper, lower, Collision(UP, depth))

    # two dynamic bodies share the correction equally
    half = depth * 0.5
    assert abs((8.5 - upper.position.y) - half) < 1e-9
    assert abs((lower.position.y - 10.0) - half) < 1e-9


def test_separate_refreshes_the_transformed_vertices_cache():
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    floor.move_to(Matrix.vector([0, 10]))
    box.move_to(Matrix.vector([0, 8.5]))
    floor.physics, box.physics = False, True
    before = box.transformed_vertices[0].y

    separate(box, floor, Collision(UP, 0.5))

    # separating through move() marks the cache dirty, so vertices follow the body
    assert abs((before - box.transformed_vertices[0].y) - 0.5) < 1e-9


def test_single_separation_resolves_a_static_contact():
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    floor.move_to(Matrix.vector([0, 10]))
    box.move_to(Matrix.vector([0, 8.5]))
    floor.physics, box.physics = False, True

    collision = detect_collision(box, floor)
    separate(box, floor, collision)

    # full projection clears the whole overlap in a single call
    residual = detect_collision(box, floor)
    assert residual is None or residual.depth < 1e-6


def test_contact_points_are_taken_before_separation():
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    floor.move_to(Matrix.vector([0, 10]))
    box.move_to(Matrix.vector([0, 8.5]))
    floor.physics, box.physics = False, True
    collision = detect_collision(box, floor)

    contact0, _ = find_contact_points(box, floor, collision)

    # the manifold reflects the overlapping config: the box top corner at y=9.5
    assert abs(contact0.y - 9.5) < 1e-6
    # and the bodies are pushed apart afterwards
    residual = detect_collision(box, floor)
    assert residual is None or residual.depth < 1e-6


def test_resting_box_does_not_sink_through_floor():
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    engine.add_body(box.move_to(Matrix.vector([0, 7])))

    for _ in range(180):
        engine.step(1 / 60)

    collision = detect_collision(box, floor)
    depth = 0.0 if collision is None else collision.depth
    # at rest the penetration stays bounded near the slop, not growing
    assert depth < 0.1
    # the box settles roughly one half-height above the floor top (y=9)
    assert 7.5 < box.position.y < 8.2
