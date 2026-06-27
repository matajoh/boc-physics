"""Tests for the colour-batched XPBD solver (xpbd_kernel)."""

import random

from bocpy import Matrix
import pytest

from bocphysics import solver, xpbd, xpbd_kernel
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind
from bocphysics.engine import PhysicsEngine
from bocphysics.physics import Physics

GRAVITY = Matrix.vector([0, 9.81])
SUB_DT = (1 / 60) / 8
FRICTION = Physics()


def make_circle(uid, x, y, vx=0.0, vy=0.0, spin=0.0, radius=1.0):
    """Build a dynamic unit circle at (x, y) with the given motion state."""
    body = Circle.create(radius, 2.0, (200, 100, 50)).move_to(Matrix.vector([x, y]))
    body.physics = True
    body.uid = uid
    body.linear_velocity = Matrix.vector([vx, vy])
    body.angular_velocity = spin
    return body


def make_disjoint_pairs(n, seed):
    """Build n far-apart overlapping circle pairs sharing no body, so one colour holds all."""
    rng = random.Random(seed)
    bodies, pairs = [], []
    for i in range(n):
        cx = i * 20.0
        a = make_circle(2 * i, cx - 0.7, 0.0, rng.uniform(1, 4), rng.uniform(-2, 2),
                        rng.uniform(-3, 3))
        b = make_circle(2 * i + 1, cx + 0.7, 0.0, rng.uniform(-4, -1), rng.uniform(-2, 2),
                        rng.uniform(-3, 3))
        bodies += [a, b]
        pairs.append((a, b))

    return bodies, pairs


def make_chain(n, seed):
    """Build a horizontal chain of n overlapping circles sharing successive endpoints."""
    rng = random.Random(seed)
    bodies = [make_circle(i, i * 1.4, 0.0, rng.uniform(-2, 2), rng.uniform(-2, 2),
                          rng.uniform(-2, 2)) for i in range(n)]
    pairs = [(bodies[i], bodies[i + 1]) for i in range(n - 1)]
    return bodies, pairs


def body_state(bodies):
    """Snapshot each body's (x, y, angle, vx, vy, spin) as plain floats."""
    return [(b.position.x, b.position.y, b.angle,
             b.linear_velocity.x, b.linear_velocity.y, b.angular_velocity)
            for b in bodies]


def assert_state_close(serial, batched):
    """Each body's pose and velocity match within a tight numeric band."""
    assert len(serial) == len(batched)
    for row_s, row_b in zip(serial, batched):
        for value_s, value_b in zip(row_s, row_b):
            assert value_b == pytest.approx(value_s, rel=1e-9, abs=1e-9)


def test_disjoint_pairs_form_one_colour():
    """Body-disjoint pairs all land in a single colour, exercising the within-colour path."""
    _bodies, pairs = make_disjoint_pairs(6, seed=1)
    constraints = xpbd.build_contacts(pairs)
    colours = xpbd_kernel.colour_contacts(constraints)
    assert len(colours) == 1
    assert len(colours[0]) == len(constraints)


def test_single_colour_batched_matches_serial():
    """Within one colour the batched kernel reproduces the serial XPBD sub-step bit-for-bit."""
    serial_bodies, serial_pairs = make_disjoint_pairs(6, seed=2)
    xpbd.solve_substep(FRICTION, serial_bodies, serial_pairs, GRAVITY, SUB_DT)

    batched_bodies, batched_pairs = make_disjoint_pairs(6, seed=2)
    xpbd_kernel.solve_substep(FRICTION, batched_bodies, batched_pairs, GRAVITY, SUB_DT)

    assert_state_close(body_state(serial_bodies), body_state(batched_bodies))


def test_colours_are_body_disjoint():
    """A shared-body chain splits into multiple colours, each internally body-disjoint."""
    _bodies, pairs = make_chain(8, seed=3)
    constraints = xpbd.build_contacts(pairs)
    colours = xpbd_kernel.colour_contacts(constraints)
    assert len(colours) > 1
    for colour in colours:
        movable = [id(c.a) for c in colour if c.a.physics] + \
                  [id(c.b) for c in colour if c.b.physics]
        assert len(movable) == len(set(movable))


def make_engine():
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900, DetectionKind.QUADTREE,
                         show_contacts=False, num_substeps=8)


def build_pile(engine):
    """Drop a non-overlapping grid of boxes onto a static floor so neither solver ejects any."""
    floor = Polygon.create_rectangle(40, 2, 2.0, (90, 90, 90), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 12])))
    for col in range(4):
        for row in range(2):
            box = Polygon.create_rectangle(2, 2, 2.0, (200, 100, 50))
            box.physics = True
            box.move_to(Matrix.vector([-4.5 + col * 3.0, 4.0 + row * 3.0]))
            engine.add_body(box)


def settle(batched, frames=250):
    """Settle the pile to rest with the batched solver on or off, returning the dynamic bodies."""
    engine = make_engine()
    build_pile(engine)
    solver.use_batched_solver = batched
    try:
        for _ in range(frames):
            engine.step(1 / 60)
    finally:
        solver.use_batched_solver = False

    return [body for body in engine.bodies if body.physics]


def kinetic_energy(bodies):
    """Total translational plus rotational kinetic energy of the bodies."""
    return sum(0.5 * b.mass * b.linear_velocity.magnitude_squared()
               + 0.5 * b.inertia * b.angular_velocity ** 2 for b in bodies)


def test_full_scene_settles_like_serial():
    """The batched XPBD solver settles the pile to the same rest as serial within a band."""
    reference = settle(False)
    batched = settle(True)

    assert len(batched) == len(reference)
    ref_top = min(body.position.y for body in reference)
    batched_top = min(body.position.y for body in batched)
    assert batched_top == pytest.approx(ref_top, abs=1.0)
    assert kinetic_energy(batched) <= kinetic_energy(reference) * 1.2 + 1e-6


def test_batched_is_deterministic():
    """Two batched runs of the same scene produce bit-identical results."""
    first = body_state(settle(True))
    second = body_state(settle(True))
    assert first == second
