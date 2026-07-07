"""Tests for the shared substep solver core."""

import math
import random

from bocpy import Matrix
import pytest

from bocphysics import physics
from bocphysics.bodies import Circle, Polygon
from bocphysics.collisions import detect_collision
from bocphysics.config import DetectionKind
from bocphysics.contacts import build_contacts, relative_normal_velocity
from bocphysics.engine import PhysicsEngine
from bocphysics.physics import Physics
from bocphysics.scene import make_pyramid_scene
from bocphysics.solver import Solver

GRAVITY = Matrix.vector([0, 9.81])
SUB_DT = (1 / 60) / 4
JACOBI_SUB_DT = (1 / 60) / 20
FRICTION = Physics()
ELASTIC = Physics(restitution=1.0, dynamic_friction=0.0)
INELASTIC = Physics(restitution=0.0)
JACOBI_PHYS = Physics(restitution=0.0, static_friction=0.5, dynamic_friction=0.5)


def make_engine(num_substeps=16) -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900,
                         DetectionKind.QUADTREE, show_contacts=False,
                         num_substeps=num_substeps)


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


def test_solver_core_matches_engine_substep():
    """The Solver core reproduces the engine's substep solve exactly."""
    positions = [Matrix.vector([-2, 0]), Matrix.vector([0, 0]), Matrix.vector([1.6, 0])]

    def build_group():
        """Build a small dynamic group at the fixed positions."""
        bodies = []
        for pos in positions:
            body = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(pos.copy())
            body.physics = True
            bodies.append(body)

        pairs = [(bodies[0], bodies[1]), (bodies[1], bodies[2])]
        return bodies, pairs

    gravity = Matrix.vector([0, 9.81])
    sub_dt = (1 / 60) / 4

    engine = make_engine()
    ref_bodies, ref_pairs = build_group()
    for body in ref_bodies:
        engine.add_body(body)
    engine.solve_substep(ref_pairs, sub_dt)

    cand_bodies, cand_pairs = build_group()
    for i, body in enumerate(cand_bodies):
        body.uid = i
    solver = Solver(cand_bodies, gravity, engine.physics)
    solver.solve(sub_dt, engine.num_substeps, cand_pairs, None)

    for r, c in zip(ref_bodies, cand_bodies):
        assert r.position.x == c.position.x
        assert r.position.y == c.position.y
        assert r.linear_velocity.x == c.linear_velocity.x
        assert r.linear_velocity.y == c.linear_velocity.y
        assert r.angular_velocity == c.angular_velocity


def test_polygon_group_core_matches_engine():
    """The core matches the engine for rotating polygon contacts too."""
    def build_group():
        """Build a two-polygon stack group."""
        a = Polygon.create_rectangle(2.0, 2.0, 2.0, (200, 100, 50))
        b = Polygon.create_rectangle(2.0, 2.0, 2.0, (50, 100, 200))
        a.physics = b.physics = True
        a.move_to(Matrix.vector([0, 0])).rotate_to(0.2)
        b.move_to(Matrix.vector([0.5, -1.8])).rotate_to(-0.1)
        return [a, b], [(a, b)]

    gravity = Matrix.vector([0, 9.81])
    sub_dt = (1 / 60) / 4

    engine = make_engine()
    ref_bodies, ref_pairs = build_group()
    for body in ref_bodies:
        engine.add_body(body)
    engine.solve_substep(ref_pairs, sub_dt)

    cand_bodies, cand_pairs = build_group()
    for i, body in enumerate(cand_bodies):
        body.uid = i
    solver = Solver(cand_bodies, gravity, engine.physics)
    solver.solve(sub_dt, engine.num_substeps, cand_pairs, None)

    for r, c in zip(ref_bodies, cand_bodies):
        assert r.position.x == c.position.x
        assert r.position.y == c.position.y
        assert r.angle == c.angle
        assert r.angular_velocity == c.angular_velocity


def make_random_body(rng: random.Random):
    """Build a random dynamic circle or polygon with random motion state."""
    kind = rng.random()
    if kind < 0.4:
        body = Circle.create(rng.uniform(0.6, 1.2), 2.0, (200, 100, 50))
    elif kind < 0.7:
        body = Polygon.create_rectangle(rng.uniform(1.2, 2.2),
                                        rng.uniform(1.2, 2.2), 2.0, (50, 120, 200))
    else:
        body = Polygon.create_regular_polygon(rng.randint(3, 6),
                                              rng.uniform(0.8, 1.3), 2.0, (180, 60, 160))

    body.physics = True
    body.move_to(Matrix.vector([rng.uniform(-12, 12), rng.uniform(-12, 6)]))
    body.rotate_to(rng.uniform(0, 6.28))
    body.linear_velocity = Matrix.vector([rng.uniform(-5, 5), rng.uniform(-5, 5)])
    body.angular_velocity = rng.uniform(-3, 3)
    return body


@pytest.mark.parametrize("seed", range(30))
def test_integrate_block_is_bit_exact_with_per_body_step(seed):
    """Batched integration must reproduce per-body step to the last bit."""
    rng = random.Random(seed)
    gravity = Matrix.vector([0, 9.81])
    dt = (1 / 60) / 4
    count = rng.randint(1, 12)

    states = [(rng.random(), rng.uniform(-12, 12), rng.uniform(-12, 6),
               rng.uniform(0, 6.28), rng.uniform(-5, 5), rng.uniform(-5, 5),
               rng.uniform(-3, 3), rng.uniform(0.6, 1.2), rng.randint(3, 6),
               rng.uniform(1.2, 2.2), rng.uniform(1.2, 2.2), rng.uniform(0.8, 1.3))
              for _ in range(count)]

    def build(state):
        """Build one dynamic body from a fixed numeric state tuple."""
        (k, x, y, ang, vx, vy, spin, r, sides, w, h, poly_r) = state
        if k < 0.4:
            body = Circle.create(r, 2.0, (200, 100, 50))
        elif k < 0.7:
            body = Polygon.create_rectangle(w, h, 2.0, (50, 120, 200))
        else:
            body = Polygon.create_regular_polygon(sides, poly_r, 2.0, (180, 60, 160))

        body.physics = True
        body.move_to(Matrix.vector([x, y])).rotate_to(ang)
        body.linear_velocity = Matrix.vector([vx, vy])
        body.angular_velocity = spin
        return body

    reference = [build(s) for s in states]
    candidate = [build(s) for s in states]

    for body in reference:
        body.step(dt, gravity)

    Solver(candidate, gravity, Physics()).integrate_block(dt)

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
            for i in range(r.transformed_vertices_block_.rows):
                assert r.transformed_vertices_block_[i, 0] == c.transformed_vertices_block_[i, 0]
                assert r.transformed_vertices_block_[i, 1] == c.transformed_vertices_block_[i, 1]


def test_integrate_block_handles_empty_region():
    """An empty body list integrates to a no-op without error."""
    Solver([], Matrix.vector([0, 9.81]), Physics()).integrate_block(1 / 60)


def test_broad_phase_pair_order_is_deterministic():
    """The broad phase emits the same pair order across repeated runs of a scene."""
    engine = make_engine()
    bodies = []
    for i in range(6):
        box = Polygon.create_rectangle(2, 2, 1.0, (1, 1, 1))
        box.move_to(Matrix.vector([i * 1.5, 10 - i * 0.7]))
        box.physics = box.collision = True
        bodies.append(box)

    first: list = []
    engine.detection.find_all_intersections(bodies, first)
    second: list = []
    engine.detection.find_all_intersections(bodies, second)

    assert [(id(a), id(b)) for a, b in first] == [(id(a), id(b)) for a, b in second]
    assert first


# --- Jacobi accumulation core --------------------------------------------------------------------


def build_pile():
    """Three stacked boxes on a static floor; returns (bodies, pairs)."""
    floor = Polygon.create_rectangle(20.0, 2.0, 2.0, (80, 80, 80), is_static=True)
    floor.physics = False
    floor.move_to(Matrix.vector([0.0, 6.0]))
    floor.uid = 0
    boxes = []
    for k in range(3):
        box = Polygon.create_rectangle(2.0, 2.0, 2.0, (50, 120, 200))
        box.physics = True
        box.move_to(Matrix.vector([0.05 * k, 4.0 - 2.0 * k]))
        box.uid = k + 1
        boxes.append(box)
    pairs = [(floor, boxes[0]), (boxes[0], boxes[1]), (boxes[1], boxes[2])]
    return boxes, pairs


def prepared_solver(boxes, pairs):
    """Integrate one sub-step and stage the frozen-pose inputs for the position pass."""
    solver = Solver(boxes, GRAVITY, JACOBI_PHYS)
    previous = solver.snapshot_poses()
    solver.integrate_block(JACOBI_SUB_DT)
    solver.constraints = build_contacts(pairs, None)
    solver.prev_pose = {id(body): pose for body, pose in zip(solver.bodies, previous)}
    return solver


def dynamic(engine):
    """Return the engine's dynamic bodies."""
    return [b for b in engine.bodies if b.physics]


def total_ke(bodies):
    """Sum translational and rotational kinetic energy."""
    energy = 0.0
    for b in bodies:
        energy += 0.5 * b.mass * b.linear_velocity.magnitude_squared()
        energy += 0.5 * b.inertia * b.angular_velocity**2
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
    engine.physics = JACOBI_PHYS
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
    body = make_circle(1, 2)
    snapshot = Solver([body], GRAVITY, Physics()).snapshot_poses()
    body.move(Matrix.vector([10, 10]))
    body.rotate_to(0.5)
    pose, = snapshot
    assert (pose[0, 0], pose[0, 1], pose[0, 2]) == (1.0, 2.0, 0.0)


def test_low_speed_restitution_is_gated_off():
    """A resting-speed approach stays gated (e = 0), so the velocity pass adds no rebound."""
    a = make_circle(0, 0)
    b = make_circle(0, 1.5, vy=-0.05)
    solver = Solver([a, b], GRAVITY, FRICTION)
    solver.constraints = build_contacts([(a, b)], None)
    assert solver.constraints
    assert abs(solver.constraints[0].bias_velocity) <= 2 * GRAVITY.magnitude() * SUB_DT

    solver.prev_pose = {id(body): pose
                        for body, pose in zip(solver.bodies, solver.snapshot_poses())}
    lambdas = solver.accumulate_positions()
    solver.apply_positions()
    solver.accumulate_velocities(lambdas, SUB_DT)
    solver.apply_velocities()

    after = relative_normal_velocity(a, b, solver.constraints[0].r_a,
                                     solver.constraints[0].r_b, solver.constraints[0].normal)
    assert after == pytest.approx(0.0, abs=1e-9)


def test_elastic_impact_reverses_relative_normal_velocity():
    """A head-on, equal-mass, perfectly elastic impact reflects the relative normal velocity."""
    a = make_circle(0, 0, vy=2.5)
    b = make_circle(0, 1.5, vy=-2.5)
    solver = Solver([a, b], GRAVITY, ELASTIC)
    solver.constraints = build_contacts([(a, b)], None)
    assert solver.constraints
    before = solver.constraints[0].bias_velocity
    assert before < 0

    solver.prev_pose = {id(body): pose
                        for body, pose in zip(solver.bodies, solver.snapshot_poses())}
    lambdas = solver.accumulate_positions()
    solver.apply_positions()
    solver.accumulate_velocities(lambdas, SUB_DT)
    solver.apply_velocities()

    after = relative_normal_velocity(a, b, solver.constraints[0].r_a,
                                     solver.constraints[0].r_b, solver.constraints[0].normal)
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
    solver = Solver([box], GRAVITY, INELASTIC)
    for _ in range(frames):
        solver.solve(SUB_DT, 4, pairs, None)
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
    floor = make_static_box(0, 10)
    ball = Circle.create(0.5, 2.0, (200, 100, 50))
    ball.physics = True
    ball.move_to(Matrix.vector([0, 0]))
    ball.linear_velocity = Matrix.vector([0, 0])
    ball.angular_velocity = 0.0
    pairs = [(ball, floor)]
    for uid, body in enumerate([floor, ball]):
        body.uid = uid
    solver = Solver([ball], GRAVITY, cfg)
    impact = rebound = 0.0
    for _ in range(frames):
        solver.solve(SUB_DT, 4, pairs, None)
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
    solver = Solver(boxes, GRAVITY, FRICTION)
    for _ in range(180):
        solver.solve(SUB_DT, 4, pairs, None)

    for box in boxes:
        assert math.isfinite(box.position.x) and math.isfinite(box.position.y)
        assert math.isfinite(box.angle)
        assert box.position.y < 13.0
