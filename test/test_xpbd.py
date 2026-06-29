"""Tests for the serial 2D XPBD contact solver (Mueller et al. 2020, Algorithm 2)."""

import math
import random

from bocpy import Matrix
import pytest

from bocphysics import xpbd
from bocphysics.bodies import Circle, Polygon
from bocphysics.collisions import detect_collision
from bocphysics.physics import Physics

GRAVITY = Matrix.vector([0, 9.81])
SUB_DT = (1 / 60) / 4
FRICTION = Physics()
ELASTIC = Physics(restitution=1.0, dynamic_friction=0.0)
INELASTIC = Physics(restitution=0.0)


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


# --- kinematic helpers ---------------------------------------------------------------------------


def test_generalized_inverse_mass_static_is_zero():
    """A static body contributes no inverse mass, keeping the effective mass finite."""
    floor = make_static_box(0, 0)
    assert xpbd.generalized_inverse_mass(floor, Matrix.vector([1, 2]), Matrix.vector([0, 1])) == 0.0


def test_generalized_inverse_mass_dynamic_matches_formula():
    """Generalised inverse mass is 1/m + (r x dir)^2 / I for a dynamic body."""
    body = make_circle(0, 0)
    body.inv_mass = 0.5
    body.inv_inertia = 0.25
    r = Matrix.vector([0, 2])
    direction = Matrix.vector([1, 0])
    # r x dir = 0*0 - 2*1 = -2, so result = 0.5 + 4 * 0.25 = 1.5.
    assert xpbd.generalized_inverse_mass(body, r, direction) == pytest.approx(1.5)


def test_contact_velocity_static_is_zero():
    """A static body's material points are at rest."""
    floor = make_static_box(0, 0)
    velocity = xpbd.contact_velocity(floor, Matrix.vector([1, 1]))
    assert velocity.x == 0.0 and velocity.y == 0.0


def test_contact_velocity_dynamic_matches_formula():
    """Material-point velocity is v + omega x r = v + omega * perpendicular(r)."""
    body = make_circle(0, 0, vx=3.0, vy=-1.0, omega=2.0)
    # perpendicular([0, 2]) = [-2, 0]; omega * that = [-4, 0]; plus v = [-1, -1].
    velocity = xpbd.contact_velocity(body, Matrix.vector([0, 2]))
    assert velocity.x == pytest.approx(-1.0)
    assert velocity.y == pytest.approx(-1.0)


def test_relative_normal_velocity_is_negative_when_approaching():
    """Two bodies closing along the normal have negative relative normal velocity."""
    a = make_circle(0, 0, vy=0.0)
    b = make_circle(0, 1.5, vy=-5.0)
    normal = Matrix.vector([0, 1])
    rel = xpbd.relative_normal_velocity(a, b, Matrix.vector([0, 0]),
                                        Matrix.vector([0, 0]), normal)
    assert rel == pytest.approx(-5.0)


# --- contact construction ------------------------------------------------------------------------


def test_build_contacts_skips_static_static_pairs():
    """A pair where neither body is dynamic produces no constraints."""
    a = make_static_box(0, 0, width=4, height=4)
    b = make_static_box(1, 0, width=4, height=4)
    assert xpbd.build_contacts([(a, b)]) == []


def test_build_contacts_skips_non_penetrating_pairs():
    """Far-apart bodies generate no constraints."""
    a = make_circle(0, 0)
    b = make_circle(50, 0)
    assert xpbd.build_contacts([(a, b)]) == []


def test_build_contacts_emits_penetrating_with_raw_bias():
    """A penetrating pair yields constraints whose bias is the raw pre-solve normal velocity."""
    a = make_circle(0, 0)
    b = make_circle(1.5, 0, vx=-4.0)
    constraints = xpbd.build_contacts([(a, b)])
    assert constraints
    for constraint in constraints:
        expected = xpbd.relative_normal_velocity(
            constraint.a, constraint.b, constraint.r_a, constraint.r_b, constraint.normal)
        assert constraint.bias_velocity == pytest.approx(expected)
        assert constraint.depth > 0


def test_build_contacts_records_overlay_points():
    """When an overlay set is given, the contact points are recorded into it."""
    a = make_circle(0, 0)
    b = make_circle(1.5, 0)
    overlay = set()
    xpbd.build_contacts([(a, b)], overlay)
    assert len(overlay) >= 1


# --- position pass -------------------------------------------------------------------------------


def test_solve_positions_returns_one_lambda_per_constraint_and_separates():
    """The position pass returns a lambda per constraint and pushes overlapping bodies apart."""
    a = make_circle(0, 0)
    b = make_circle(1.5, 0)
    constraints = xpbd.build_contacts([(a, b)])
    before = b.position.x - a.position.x
    lambdas = xpbd.solve_positions(constraints)
    assert len(lambdas) == len(constraints)
    assert all(lam > 0 for lam in lambdas)
    assert (b.position.x - a.position.x) > before


# --- velocity halves -----------------------------------------------------------------------------


def test_snapshot_poses_is_alias_safe():
    """A snapshot stores scalars, so moving the body afterwards never mutates it."""
    body = make_circle(1, 2)
    snapshot = xpbd.snapshot_poses([body])
    body.move(Matrix.vector([10, 10]))
    body.rotate_to(0.5)
    assert snapshot == [(1.0, 2.0, 0.0)]


def test_derive_velocities_from_pose_delta():
    """Velocity is reconstructed as the pose delta over the sub-step."""
    body = make_circle(0, 0)
    previous = xpbd.snapshot_poses([body])
    body.move(Matrix.vector([0.1, 0.2]))
    body.rotate_to(0.05)
    xpbd.derive_velocities([body], previous, SUB_DT)
    assert body.linear_velocity.x == pytest.approx(0.1 / SUB_DT)
    assert body.linear_velocity.y == pytest.approx(0.2 / SUB_DT)
    assert body.angular_velocity == pytest.approx(0.05 / SUB_DT)


def test_low_speed_restitution_is_gated_off():
    """A resting-speed approach stays gated (e = 0), so the velocity pass adds no rebound."""
    a = make_circle(0, 0)
    b = make_circle(0, 1.5, vy=-0.05)
    constraints = xpbd.build_contacts([(a, b)])
    assert constraints
    assert abs(constraints[0].bias_velocity) <= 2 * GRAVITY.magnitude() * SUB_DT
    lambdas = xpbd.solve_positions(constraints)
    xpbd.solve_velocities(FRICTION, constraints, lambdas, SUB_DT, GRAVITY)
    after = xpbd.relative_normal_velocity(
        a, b, constraints[0].r_a, constraints[0].r_b, constraints[0].normal)
    assert after == pytest.approx(0.0, abs=1e-9)


def test_elastic_impact_reverses_relative_normal_velocity():
    """A head-on, equal-mass, perfectly elastic impact reflects the relative normal velocity."""
    a = make_circle(0, 0, vy=2.5)
    b = make_circle(0, 1.5, vy=-2.5)
    constraints = xpbd.build_contacts([(a, b)])
    assert constraints
    before = constraints[0].bias_velocity
    assert before < 0
    lambdas = xpbd.solve_positions(constraints)
    xpbd.solve_velocities(ELASTIC, constraints, lambdas, SUB_DT, GRAVITY)
    after = xpbd.relative_normal_velocity(
        a, b, constraints[0].r_a, constraints[0].r_b, constraints[0].normal)
    assert after == pytest.approx(-before, abs=1e-6)


# --- end-to-end behaviour ------------------------------------------------------------------------


def run_drop(frames=180):
    """Drop one dynamic box onto a static floor and return the settled box."""
    floor = make_static_box(0, 10)
    box = Polygon.create_rectangle(2.0, 2.0, 1.0, (50, 100, 200))
    box.physics = True
    box.move_to(Matrix.vector([0, 0]))
    box.linear_velocity = Matrix.vector([0, 0])
    box.angular_velocity = 0.0
    pairs = [(box, floor)]
    for uid, body in enumerate([floor, box]):
        body.uid = uid
    for _ in range(frames):
        xpbd.solve_group_substep(INELASTIC, [box], pairs, GRAVITY, SUB_DT, 4)
    return box, floor


def test_solve_group_substep_settles_box_on_floor():
    """A dropped box comes to rest on the floor without tunnelling or jitter."""
    box, floor = run_drop()
    assert math.isfinite(box.position.x) and math.isfinite(box.position.y)
    assert abs(box.position.x) < 0.5
    assert box.position.y < 10.0
    assert abs(box.linear_velocity.y) < 0.5
    collision = detect_collision(box, floor)
    assert collision is None or collision.depth < 0.1


def test_solver_is_deterministic():
    """The same scene run twice produces bit-identical final state."""
    box_a, _ = run_drop(frames=60)
    box_b, _ = run_drop(frames=60)
    assert box_a.position.x == box_b.position.x
    assert box_a.position.y == box_b.position.y
    assert box_a.linear_velocity.x == box_b.linear_velocity.x
    assert box_a.linear_velocity.y == box_b.linear_velocity.y
    assert box_a.angular_velocity == box_b.angular_velocity


def measured_restitution(restitution, frames=150):
    """Drop a ball and return its empirical coefficient of restitution (rebound / impact speed)."""
    physics = Physics(restitution=restitution, dynamic_friction=0.0)
    floor = make_static_box(0, 10)
    ball = Circle.create(0.5, 2.0, (200, 100, 50))
    ball.physics = True
    ball.move_to(Matrix.vector([0, 0]))
    ball.linear_velocity = Matrix.vector([0, 0])
    ball.angular_velocity = 0.0
    pairs = [(ball, floor)]
    impact = rebound = 0.0
    for _ in range(frames):
        xpbd.solve_group_substep(physics, [ball], pairs, GRAVITY, SUB_DT, 4)
        vy = ball.linear_velocity.y
        impact = max(impact, vy)
        rebound = max(rebound, -vy)
    return rebound / impact if impact else 0.0


@pytest.mark.parametrize("restitution", [0.0, 0.5, 0.9])
def test_restitution_coefficient_tracks_configured_e(restitution):
    """A dropped ball's measured restitution tracks the configured coefficient."""
    assert measured_restitution(restitution) == pytest.approx(restitution, abs=0.1)


@pytest.mark.parametrize("seed", range(20))
def test_random_pile_stays_finite_and_bounded(seed):
    """A non-overlapping column of boxes settles without NaN/inf or escaping the floor."""
    rng = random.Random(seed)
    floor = make_static_box(0, 10)
    boxes = []
    for level in range(rng.randint(2, 4)):
        box = Polygon.create_rectangle(2.0, 2.0, 1.0, (50, 100, 200))
        box.physics = True
        box.move_to(Matrix.vector([rng.uniform(-0.3, 0.3), -3.0 * level]))
        box.linear_velocity = Matrix.vector([0, 0])
        box.angular_velocity = 0.0
        boxes.append(box)

    pairs = [(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
    pairs += [(box, floor) for box in boxes]
    for uid, body in enumerate([floor] + boxes):
        body.uid = uid
    for _ in range(180):
        xpbd.solve_group_substep(FRICTION, boxes, pairs, GRAVITY, SUB_DT, 4)

    for box in boxes:
        assert math.isfinite(box.position.x) and math.isfinite(box.position.y)
        assert math.isfinite(box.angle)
        assert box.position.y < 13.0
