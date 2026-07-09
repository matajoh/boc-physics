"""Tests for the shared substep solver core."""

import itertools
import math
import random
from typing import Union

from bocpy import Matrix
import pytest

from bocphysics import physics
from bocphysics.bodies import Circle, Polygon, RigidBody, RigidBodyState
from bocphysics.collisions import detect_collision
from bocphysics.config import DetectionKind
from bocphysics.contacts import build_contacts, relative_normal_velocity
from bocphysics.engine import PhysicsEngine
from bocphysics.physics import Physics
from bocphysics.scene import make_pyramid_scene
from bocphysics.solver import Solver

GRAVITY = Matrix.vector([0, 9.81])
NUM_SUBSTEPS = 20
DT = 1 / 60
SUB_DT = DT / NUM_SUBSTEPS
FRICTION = Physics()
ELASTIC = Physics(restitution=1.0, dynamic_friction=0.0)
INELASTIC = Physics(restitution=0.0)
PHYS = Physics(restitution=0.0, static_friction=0.5, dynamic_friction=0.5)
UID = itertools.count(start=0)


def make_engine(num_substeps=16) -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900,
                         DetectionKind.QUADTREE, show_contacts=False,
                         num_substeps=num_substeps)


def make_circle(x: float, y: float, angle=0, vx=0.0, vy=0.0, omega=0.0, radius=1.0, density=2.0, is_static=False):
    """Build a dynamic circle at (x, y) with the given motion state."""
    body = Circle.create(radius, density, (127, 127, 127), is_static)
    body.physics = not is_static
    body.collision = True
    body.move_to(Matrix.vector([x, y])).rotate_to(angle)
    body.linear_velocity[:] = vx, vy
    body.angular_velocity.x = omega
    body.uid = next(UID)
    return body


def make_polygon(num_sides: int, x, y, angle=0, vx=0.0, vy=0.0, omega=0.0, radius=1.0, density=2.0, is_static=False):
    """Build a static rectangle floor centred at (x, y)."""
    body = Polygon.create_regular_polygon(num_sides, radius, density, (127, 127, 127), is_static)
    body.physics = not is_static
    body.collision = True
    body.move_to(Matrix.vector([x, y])).rotate_to(angle)
    body.linear_velocity[:] = vx, vy
    body.angular_velocity.x = omega
    body.uid = next(UID)
    return body


def make_box(x, y, angle=0, width=2, height=2, vx=0.0, vy=0.0, omega=0.0, is_static=False):
    """Build a static rectangle floor centred at (x, y)."""
    body = Polygon.create_rectangle(width, height, 1.0, (127, 127, 127), is_static)
    body.physics = not is_static
    body.collision = True
    body.move_to(Matrix.vector([x, y])).rotate_to(angle)
    body.uid = next(UID)

    if not is_static:
        body.linear_velocity[:] = vx, vy
        body.angular_velocity.x = omega

    return body


def make_random_body(rng: random.Random):
    """Build a random dynamic circle or polygon with random motion state."""
    kind = rng.random()
    x = rng.uniform(-12, 12)
    y = rng.uniform(-12, 6)
    angle = rng.uniform(0, 6.28)
    vx = rng.uniform(-5, 5)
    vy = rng.uniform(-5, 5)
    omega = rng.uniform(-3, 3)
    if kind < 0.4:
        radius = rng.uniform(0.6, 1.2)
        body = make_circle(x, y, angle, vx, vy, omega, radius)
    elif kind < 0.7:
        width = rng.uniform(1.2, 2.2)
        height = rng.uniform(1.2, 2.2)
        body = make_box(x, y, angle, width, height, vx, vy, omega)
    else:
        num_sides = rng.randint(3, 6)
        radius = rng.uniform(0.8, 1.3)
        body = make_polygon(num_sides, x, y, angle, vx, vy, omega, radius)

    return body


def to_state(body: RigidBody) -> RigidBodyState:
    return RigidBodyState(body.state)


def test_solver_core_matches_engine_substep():
    """The Solver core reproduces the engine's substep solve exactly."""
    def build_group(as_state: bool):
        """Build a small dynamic group at the fixed positions."""
        bodies = []
        for x, y in [(-2, 0), (0, 0), (1.6, 0)]:
            body = make_circle(x, y)
            if as_state:
                body = to_state(body)

            bodies.append(body)

        pairs = [(bodies[0], bodies[1]), (bodies[1], bodies[2])]
        return bodies, pairs

    engine = make_engine()
    ref_bodies, ref_pairs = build_group(False)
    for body in ref_bodies:
        engine.add_body(body)
    engine.solve_substep(DT, ref_pairs)

    cand_bodies, cand_pairs = build_group(True)
    solver = Solver(cand_bodies, GRAVITY, engine.physics)
    solver.solve(DT, engine.num_substeps, cand_pairs, None)

    for r, c in zip(ref_bodies, cand_bodies):
        assert r.position.x == c.position.x
        assert r.position.y == c.position.y
        assert r.linear_velocity.x == c.linear_velocity.x
        assert r.linear_velocity.y == c.linear_velocity.y
        assert r.angular_velocity == c.angular_velocity


RBS = Union[RigidBody, RigidBodyState]


def test_polygon_group_core_matches_engine():
    """The core matches the engine for rotating polygon contacts too."""

    def build_group(as_state: bool):
        """Build a two-polygon stack group."""
        a = make_box(0, 0, 0.2)
        b = make_box(0.5, -1.8, -0.1)
        if as_state:
            a = to_state(a)
            b = to_state(b)

        return [a, b], [(a, b)]

    engine = make_engine()
    ref_bodies, ref_pairs = build_group(False)
    for body in ref_bodies:
        engine.add_body(body)

    ref_pairs = [(engine.states[a.uid], engine.states[b.uid]) for a, b in ref_pairs]
    engine.solve_substep(DT, ref_pairs)

    cand_bodies, cand_pairs = build_group(True)
    solver = Solver(cand_bodies, GRAVITY, engine.physics)
    solver.solve(DT, engine.num_substeps, cand_pairs, None)

    for r, c in zip(ref_bodies, cand_bodies):
        assert r.position.x == c.position.x
        assert r.position.y == c.position.y
        assert r.angle == c.angle
        assert r.angular_velocity == c.angular_velocity


@pytest.mark.parametrize("seed", range(30))
def test_integrate_block_is_bit_exact_with_per_body_step(seed):
    """Batched integration must reproduce per-body step to the last bit."""
    rng = random.Random(seed)
    count = rng.randint(1, 12)

    rng = random.Random(seed)
    reference = [make_random_body(rng) for _ in range(count)]

    rng = random.Random(seed)
    candidate = [to_state(make_random_body(rng)) for _ in range(count)]

    for body in reference:
        body.step(SUB_DT, GRAVITY)

    Solver(candidate, GRAVITY, Physics()).integrate_block(SUB_DT)

    for r, c in zip(reference, candidate):
        assert r.position.x == c.position.x
        assert r.position.y == c.position.y
        assert r.linear_velocity.x == c.linear_velocity.x
        assert r.linear_velocity.y == c.linear_velocity.y
        assert r.angle == c.angle
        assert r.update_needed_ == c.update_needed_
        if isinstance(r, Polygon):
            r.update_transform()
            c.update_transform()
            for i in range(r.transformed_vertices_.rows):
                assert r.transformed_vertices_[i, 0] == c.transformed_vertices_[i, 0]
                assert r.transformed_vertices_[i, 1] == c.transformed_vertices_[i, 1]


def test_integrate_block_handles_empty_region():
    """An empty body list integrates to a no-op without error."""
    Solver([], Matrix.vector([0, 9.81]), Physics()).integrate_block(1 / 60)


def test_broad_phase_pair_order_is_deterministic():
    """The broad phase emits the same pair order across repeated runs of a scene."""
    engine = make_engine()
    bodies = []
    for i in range(6):
        x = i * 1.5
        y = 10 - i * 0.7
        box = make_box(x, y, 0, 2, 2)
        bodies.append(box)

    first: list = []
    engine.detection.find_all_intersections(bodies, first)
    second: list = []
    engine.detection.find_all_intersections(bodies, second)

    assert [(id(a), id(b)) for a, b in first] == [(id(a), id(b)) for a, b in second]
    assert first


def build_pile():
    """Three stacked boxes on a static floor; returns (bodies, pairs)."""
    floor = make_box(0, 6, 0, 20, 2, is_static=True)
    boxes = []
    for k in range(3):
        x = 0.05 * k
        y = 4.0 - 2.0 * k
        box = make_box(x, y)
        boxes.append(to_state(box))

    pairs = [(floor, boxes[0]), (boxes[0], boxes[1]), (boxes[1], boxes[2])]
    return boxes, pairs


def prepared_solver(boxes, pairs):
    """Integrate one sub-step and stage the frozen-pose inputs for the position pass."""
    solver = Solver(boxes, GRAVITY, PHYS)
    previous = solver.snapshot_poses()
    solver.integrate_block(SUB_DT)
    solver.constraints = build_contacts(pairs, None)
    solver.prev_pose = {body.uid: pose for body, pose in zip(solver.bodies, previous)}
    return solver


def dynamic(engine):
    """Return the engine's dynamic bodies."""
    return [b for b in engine.bodies if b.physics]


def total_ke(bodies):
    """Sum translational and rotational kinetic energy."""
    energy = 0.0
    for b in bodies:
        energy += 0.5 * b.mass * b.linear_velocity.magnitude_squared()
        energy += 0.5 * b.inertia * b.angular_velocity.magnitude_squared()
    return energy


def test_accumulate_positions_is_order_independent():
    """Jacobi accumulation over a frozen pose is independent of constraint order."""
    boxes, pairs = build_pile()
    solver = prepared_solver(boxes, pairs)
    assert solver.constraints

    solver.accumulate_positions()
    forward = solver.pos_acc.copy()
    solver.constraints = list(reversed(solver.constraints))
    solver.accumulate_positions()
    reverse = solver.pos_acc.copy()

    for i in range(len(boxes)):
        assert forward[i, 3] == reverse[i, 3]
        for column in range(3):
            assert abs(forward[i, column] - reverse[i, column]) < 1e-9


def test_accumulator_is_a_block_indexed_by_row():
    """The accumulator is an (N x 4) block and records a positive contribution count."""
    boxes, pairs = build_pile()
    solver = prepared_solver(boxes, pairs)

    solver.accumulate_positions()
    acc = solver.pos_acc

    assert (acc.rows, acc.columns) == (len(boxes), physics.ACC_WIDTH)
    assert sum(acc[i, 3] for i in range(len(boxes))) > 0


def test_jacobi_settles_a_stack():
    """A short box stack settles under the Jacobi solver without exploding."""
    engine = make_engine(num_substeps=20)
    engine.physics = PHYS
    for body in make_pyramid_scene(2).build():
        engine.add_body(body)

    for _ in range(300):
        engine.step(1 / 60)

    bodies = dynamic(engine)
    for body in bodies:
        assert math.isfinite(body.position.x)
        assert math.isfinite(body.position.y)
        assert abs(body.position.x) < 50.0
    assert total_ke(bodies) < 1.0


# --- velocity halves -----------------------------------------------------------------------------


def test_snapshot_poses_is_alias_safe():
    """A snapshot copies the pose, so moving the body afterwards never mutates it."""
    body = make_circle(1.0, 2.0)
    snapshot = Solver([body], GRAVITY, Physics()).snapshot_poses()
    body.move(Matrix.vector([10, 10]))
    body.rotate_to(0.5)
    pose, = snapshot
    assert (pose[0, 0], pose[0, 1], pose[0, 2]) == (1.0, 2.0, 0.0)


def test_elastic_impact_reverses_relative_normal_velocity():
    """A head-on, equal-mass, perfectly elastic impact reflects the relative normal velocity."""
    a = make_circle(0, 0, vy=2.5)
    b = make_circle(0, 1.5, vy=-2.5)
    solver = Solver([a, b], GRAVITY, ELASTIC)

    previous = solver.snapshot_poses()
    solver.integrate_block(SUB_DT)
    solver.constraints = build_contacts([(a, b)], None)
    assert solver.constraints
    before = solver.constraints[0].bias_velocity
    assert before < 0
    solver.prev_pose = {body.uid: pose
                        for body, pose in zip(solver.bodies, previous)}
    lambdas = solver.accumulate_positions()
    solver.apply_positions()
    Physics.derive_velocities(solver.bodies, previous, SUB_DT)
    solver.accumulate_velocities(lambdas, SUB_DT)
    solver.apply_velocities()

    after = relative_normal_velocity(a, b, solver.constraints[0].r_a,
                                     solver.constraints[0].r_b, solver.constraints[0].normal)
    assert after == pytest.approx(-before, abs=1e-6)


# --- end-to-end behaviour ------------------------------------------------------------------------


def run_drop(frames=180):
    """Drop one dynamic box onto a static floor and return the settled box."""
    floor = make_box(0, 10, width=40, is_static=True)
    box = to_state(make_box(0, 0))
    pairs = [(box, floor)]
    solver = Solver([box], GRAVITY, INELASTIC)
    for _ in range(frames):
        solver.solve(DT, NUM_SUBSTEPS, pairs, None)
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
    cfg = Physics(restitution=restitution, dynamic_friction=0.0)
    floor = make_box(0, 10, width=40, is_static=True)
    ball = make_circle(0, 0, radius=0.5)
    ball_state = to_state(ball)
    pairs = [(ball_state, floor)]
    solver = Solver([ball_state], GRAVITY, cfg)
    impact = rebound = 0.0
    for _ in range(frames):
        solver.solve(DT, NUM_SUBSTEPS, pairs, None)
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
    floor = make_box(0, 10, width=40, is_static=True)
    boxes = []
    for level in range(rng.randint(2, 4)):
        x = rng.uniform(-0.3, 0.3)
        y = -3.0 * level
        box = make_box(x, y)
        boxes.append(to_state(box))

    pairs = [(boxes[i], boxes[j]) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
    pairs += [(box, floor) for box in boxes]

    solver = Solver(boxes, GRAVITY, FRICTION)
    for _ in range(180):
        solver.solve(DT, NUM_SUBSTEPS, pairs, None)

    for box in boxes:
        assert math.isfinite(box.position.x) and math.isfinite(box.position.y)
        assert math.isfinite(box.angle.x)
        assert box.position.y < 13.0
