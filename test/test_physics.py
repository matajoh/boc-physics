"""Tests for contact-point generation and resting-contact integration."""

from bocpy import Matrix

from bocphysics.bodies import Polygon
from bocphysics.collisions import detect_collision
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.contacts import find_contact_points
from bocphysics.engine import PhysicsEngine


def make_engine() -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                         DetectionKind.QUADTREE, show_contacts=False)


def test_find_contact_points_is_pure_and_takes_overlapping_config():
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    floor.move_to(Matrix.vector([0, 10]))
    box.move_to(Matrix.vector([0, 8.5]))
    floor.physics, box.physics = False, True
    collision = detect_collision(box, floor)

    contact0, _, _, _ = find_contact_points(box, floor, collision)

    assert abs(contact0.y - 9.5) < 1e-6
    residual = detect_collision(box, floor)
    assert residual is not None and abs(residual.depth - collision.depth) < 1e-9


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
    assert depth < 0.1
    assert 7.5 < box.position.y < 8.2
