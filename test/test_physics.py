"""Tests for the position-correction step in the contact solver."""

from bocpy import Matrix
import pytest

from bocphysics import solver
from bocphysics.bodies import Polygon
from bocphysics.collisions import Collision, detect_collision
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.contacts import find_contact_points, separate
from bocphysics.engine import PhysicsEngine
from bocphysics.physics import Constraint, Physics, ZERO_VEC

UP = Matrix.vector([0, 1])


def make_moving_body(vy: float) -> Polygon:
    """Create a dynamic box moving along y with no spin, for bias tests."""
    body = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    body.linear_velocity = Matrix.vector([0, vy])
    body.angular_velocity = 0.0
    body.physics = True
    return body


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


def test_find_contact_points_is_pure_and_takes_overlapping_config():
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
    floor.move_to(Matrix.vector([0, 10]))
    box.move_to(Matrix.vector([0, 8.5]))
    floor.physics, box.physics = False, True
    collision = detect_collision(box, floor)

    contact0, _, _, _ = find_contact_points(box, floor, collision)

    # the manifold reflects the overlapping config: the box top corner at y=9.5
    assert abs(contact0.y - 9.5) < 1e-6
    # the query is pure: it does NOT move the bodies, so the overlap is unchanged
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
    # at rest the penetration stays bounded near the slop, not growing
    assert depth < 0.1
    # the box settles roughly one half-height above the floor top (y=9)
    assert 7.5 < box.position.y < 8.2


def test_restitution_for_applies_above_threshold():
    """A fast closing contact gets the full restitution coefficient."""
    physics = Physics(PhysicsMode.FRICTION)
    # closing at 3 m/s (> 1 m/s threshold) bounces with full restitution
    assert physics.restitution_for(-3.0) == 0.5


def test_restitution_for_zero_below_threshold():
    """A resting contact below the threshold gets zero restitution."""
    physics = Physics(PhysicsMode.FRICTION)
    # closing at 0.5 m/s (< 1 m/s threshold) settles instead of bouncing
    assert physics.restitution_for(-0.5) == 0.0


def test_restitution_bias_applies_on_fast_impact():
    """A fast impact captures a positive velocity target of -e * vn0."""
    physics = Physics(PhysicsMode.FRICTION)
    a = make_moving_body(0.0)
    b = make_moving_body(-3.0)
    # vn0 = -3 m/s; target = -0.5 * -3 = 1.5 m/s of bounce-back
    bias = physics.restitution_bias(a, b, UP, ZERO_VEC, ZERO_VEC)
    assert bias == pytest.approx(1.5)


def test_restitution_bias_zero_for_resting_contact():
    """A slow closing contact below the threshold captures no bias."""
    physics = Physics(PhysicsMode.FRICTION)
    a = make_moving_body(0.0)
    b = make_moving_body(-0.5)
    # closing at 0.5 m/s is below the 1 m/s threshold, so the target is zero
    assert physics.restitution_bias(a, b, UP, ZERO_VEC, ZERO_VEC) == 0.0


def test_restitution_bias_zero_for_separating_contact():
    """A separating contact never gets a restitution bias."""
    physics = Physics(PhysicsMode.FRICTION)
    a = make_moving_body(0.0)
    b = make_moving_body(2.0)
    # the bodies are moving apart, so there is nothing to bounce
    assert physics.restitution_bias(a, b, UP, ZERO_VEC, ZERO_VEC) == 0.0


def make_two_box_constraint(physics, upper_vel, upper_spin, lower_vel, lower_spin):
    """Build a 2-contact constraint from two overlapping boxes with given motion."""
    lower = Polygon.create_rectangle(4, 2, 3.0, (200, 100, 50))
    upper = Polygon.create_rectangle(4, 2, 2.0, (50, 100, 200))
    lower.move_to(Matrix.vector([0, 10]))
    upper.move_to(Matrix.vector([0, 8.3]))
    lower.physics = upper.physics = True
    upper.linear_velocity = Matrix.vector(list(upper_vel))
    upper.angular_velocity = upper_spin
    lower.linear_velocity = Matrix.vector(list(lower_vel))
    lower.angular_velocity = lower_spin
    collision = detect_collision(upper, lower)
    contact0, contact1, _i0, _i1 = find_contact_points(upper, lower, collision)
    constraint = physics.prepare_collision(upper, lower, collision, contact0, contact1)
    return constraint, upper, lower


def test_apply_accumulated_releases_earlier_overpush():
    """A seeded running normal total is lowered by a later separating sweep."""
    physics = Physics(PhysicsMode.FRICTION)
    constraint, _upper, _lower = make_two_box_constraint(
        physics, (0.0, -2.0), 0.0, (0.0, 1.0), 0.0)
    # seed each contact as if a prior sweep over-pushed; bodies now separate
    lam_n = [5.0, 5.0]
    lam_t = [0.0, 0.0]
    tangent_data = solver.build_tangent_data(constraint)

    physics.apply_accumulated(constraint, lam_n, lam_t, tangent_data)

    # accumulation lowers the running total -- impossible under per-iteration clamping
    assert 0.0 < lam_n[0] < 5.0
    assert 0.0 < lam_n[1] < 5.0


def test_accumulated_friction_sticks_within_static_cone():
    """A small tangential drift stays inside the static cone (stick, not capped)."""
    physics = Physics(PhysicsMode.FRICTION)._replace(static_friction=0.5,
                                                     dynamic_friction=0.3)
    constraint, _upper, _lower = make_two_box_constraint(
        physics, (0.2, 2.0), 0.0, (0.0, -1.0), 0.0)
    nc = len(constraint.contacts)
    lam_n = [0.0] * nc
    lam_t = [0.0] * nc
    tangent_data = solver.build_tangent_data(constraint)

    for _ in range(5):
        physics.apply_accumulated(constraint, lam_n, lam_t, tangent_data)

    for i in range(nc):
        # strictly inside the static cone and non-zero -- friction is sticking
        assert 0.0 < abs(lam_t[i]) < physics.static_friction * lam_n[i]


def test_accumulated_friction_slides_capped_at_kinetic():
    """A large tangential slide is capped at the kinetic cone, not the static one."""
    physics = Physics(PhysicsMode.FRICTION)._replace(static_friction=0.5,
                                                     dynamic_friction=0.3)
    constraint, _upper, _lower = make_two_box_constraint(
        physics, (40.0, 2.0), 0.0, (0.0, -1.0), 0.0)
    nc = len(constraint.contacts)
    lam_n = [0.0] * nc
    lam_t = [0.0] * nc
    tangent_data = solver.build_tangent_data(constraint)

    physics.apply_accumulated(constraint, lam_n, lam_t, tangent_data)

    for i in range(nc):
        # past the static cone -> running total pinned to the kinetic bound
        assert abs(lam_t[i]) == pytest.approx(physics.dynamic_friction * lam_n[i])


def test_accumulated_friction_zero_tangential_velocity_is_noop():
    """Pure normal closing motion leaves the tangent total at zero, no guard needed."""
    physics = Physics(PhysicsMode.FRICTION)._replace(static_friction=0.5,
                                                     dynamic_friction=0.3)
    constraint, _upper, _lower = make_two_box_constraint(
        physics, (0.0, 2.0), 0.0, (0.0, -2.0), 0.0)
    nc = len(constraint.contacts)
    lam_n = [0.0] * nc
    lam_t = [0.0] * nc
    tangent_data = solver.build_tangent_data(constraint)

    physics.apply_accumulated(constraint, lam_n, lam_t, tangent_data)

    # the fixed-axis friction applies ~0 at zero tangential speed without a guard
    assert lam_n[0] > 0.0
    assert lam_t == [0.0] * nc


def test_apply_accumulated_zero_contacts_is_noop():
    """A constraint with no contacts is a no-op, not an error."""
    physics = Physics(PhysicsMode.FRICTION)
    a = make_moving_body(1.0)
    b = make_moving_body(-1.0)
    constraint = Constraint(PhysicsMode.FRICTION, a, b, UP, ())
    before = (a.linear_velocity.x, a.linear_velocity.y, b.linear_velocity.x)

    physics.apply_accumulated(constraint, [], [], solver.build_tangent_data(constraint))

    after = (a.linear_velocity.x, a.linear_velocity.y, b.linear_velocity.x)
    assert before == after


def test_apply_accumulated_matches_probe_oracle():
    """The in-engine sweep reproduces the validated probe's result bit-for-bit.

    The oracle was captured once from .copilot/pyramid_twocoef.py (the validated
    two-coefficient probe) on this exact fixture; that scratch is gitignored, so
    the reference lives here. A regression to point-by-point Gauss-Seidel within a
    manifold (inline scatter) would shift these by orders of magnitude.
    """
    physics = Physics(PhysicsMode.FRICTION)._replace(static_friction=0.5,
                                                     dynamic_friction=0.3)
    constraint, upper, lower = make_two_box_constraint(
        physics, (1.5, 2.0), 0.3, (-0.4, -1.0), -0.1)
    lam_n = [0.0] * len(constraint.contacts)
    lam_t = [0.0] * len(constraint.contacts)
    tangent_data = solver.build_tangent_data(constraint)

    for _ in range(5):
        physics.apply_accumulated(constraint, lam_n, lam_t, tangent_data)

    assert upper.linear_velocity.x == 0.724706333583282
    assert upper.linear_velocity.y == -0.22682108127790357
    assert upper.angular_velocity == 0.26255507291734226
    assert lower.linear_velocity.x == 0.11686244427781181
    assert lower.linear_velocity.y == 0.48454738751860227
    assert lower.angular_velocity == 0.4521629778851396
