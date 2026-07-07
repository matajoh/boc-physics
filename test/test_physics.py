"""Tests for contact-point generation and resting-contact integration."""

from bocpy import Matrix
import pytest

from bocphysics.bodies import Circle, Polygon
from bocphysics.collisions import detect_collision
from bocphysics.config import DetectionKind
from bocphysics.contacts import find_contact_points
from bocphysics.engine import PhysicsEngine
from bocphysics.geometry import GeometryPool
from bocphysics.physics import Physics
from bocphysics.solver import Solver

GRAVITY = Matrix.vector([0, 9.81])
SUB_DT = (1 / 60) / 4


def make_circle(x, y, vx=0.0, vy=0.0, omega=0.0, radius=1.0):
    """Build a dynamic circle at (x, y) with the given motion state."""
    body = Circle.create(radius, 2.0, (200, 100, 50))
    body.physics = True
    body.move_to(Matrix.vector([x, y]))
    body.linear_velocity = Matrix.vector([vx, vy])
    body.angular_velocity = omega
    return body


def make_static_box(x, y, width=40.0, height=2.0):
    """Build a static rectangle floor centred at (x, y)."""
    floor = Polygon.create_rectangle(width, height, 1.0, (90, 90, 90), is_static=True)
    floor.move_to(Matrix.vector([x, y]))
    floor.physics = False
    return floor


def make_engine() -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900,
                         DetectionKind.QUADTREE, show_contacts=False)


def test_find_contact_points_is_pure_and_takes_overlapping_config():
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    floor.move_to(Matrix.vector([0, 10]))
    box.move_to(Matrix.vector([0, 8.5]))
    floor.physics, box.physics = False, True
    floor.uid, box.uid = 1, 2
    collision = detect_collision(box, floor)

    contact0, _, _, _ = find_contact_points(box, floor, collision,
                                            GeometryPool([box, floor]))

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


# --- kinematic helpers ---------------------------------------------------------------------------


def test_generalized_inverse_mass_static_is_zero():
    """A static body contributes no inverse mass, keeping the effective mass finite."""
    floor = make_static_box(0, 0)
    assert Physics.inv_mass(floor, Matrix.vector([1, 2]), Matrix.vector([0, 1])) == 0.0


def test_generalized_inverse_mass_dynamic_matches_formula():
    """Generalised inverse mass is 1/m + (r x dir)^2 / I for a dynamic body."""
    body = make_circle(0, 0)
    body.inv_mass = 0.5
    body.inv_inertia = 0.25
    r = Matrix.vector([0, 2])
    direction = Matrix.vector([1, 0])
    # r x dir = 0*0 - 2*1 = -2, so result = 0.5 + 4 * 0.25 = 1.5.
    assert Physics.inv_mass(body, r, direction) == pytest.approx(1.5)


def test_derive_velocities_from_pose_delta():
    """Velocity is reconstructed as the pose delta over the sub-step."""
    body = make_circle(0, 0)
    previous = Solver([body], GRAVITY, Physics()).snapshot_poses()
    body.move(Matrix.vector([0.1, 0.2]))
    body.rotate_to(0.05)
    Physics.derive_velocities([body], previous, SUB_DT)
    assert body.linear_velocity.x == pytest.approx(0.1 / SUB_DT)
    assert body.linear_velocity.y == pytest.approx(0.2 / SUB_DT)
    assert body.angular_velocity == pytest.approx(0.05 / SUB_DT)
