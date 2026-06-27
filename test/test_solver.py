"""Tests for the shared substep solver core."""

import random

from bocpy import Matrix
import pytest

from bocphysics import solver, xpbd
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine


def make_engine() -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                         DetectionKind.QUADTREE, show_contacts=False)


def test_solver_core_matches_engine_substep():
    """The free-function core reproduces the engine's substep solve exactly."""
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
    engine.solve_substep(ref_bodies, ref_pairs, sub_dt)

    cand_bodies, cand_pairs = build_group()
    xpbd.solve_group_substep(engine.physics, cand_bodies, cand_pairs,
                             gravity, sub_dt, engine.num_substeps, None)

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
    engine.solve_substep(ref_bodies, ref_pairs, sub_dt)

    cand_bodies, cand_pairs = build_group()
    xpbd.solve_group_substep(engine.physics, cand_bodies, cand_pairs,
                             gravity, sub_dt, engine.num_substeps, None)

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

    solver.integrate_block(candidate, gravity, dt)

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
    solver.integrate_block([], Matrix.vector([0, 9.81]), 1 / 60)


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
