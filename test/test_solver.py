"""Tests for the shared substep solver core."""

import random

from bocpy import Matrix
import pytest

from bocphysics import solver
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine
from bocphysics.physics import Constraint

UP = Matrix.vector([0, 1])


def make_engine() -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                         DetectionKind.QUADTREE, show_contacts=False)


def test_build_manifold_drops_false_positive():
    """Two far-apart bodies yield no manifold."""
    a = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0, 0]))
    b = Circle.create(1.0, 2.0, (50, 100, 200)).move_to(Matrix.vector([50, 0]))
    a.physics = b.physics = True
    assert solver.build_manifold(a, b, None) is None


def test_build_manifold_records_contacts_when_set_given():
    """A real contact records its points into the supplied set."""
    a = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0, 0]))
    b = Circle.create(1.0, 2.0, (50, 100, 200)).move_to(Matrix.vector([1.5, 0]))
    a.physics = b.physics = True
    contacts = set()
    manifold = solver.build_manifold(a, b, contacts)
    assert manifold is not None
    assert len(contacts) >= 1


def test_build_group_manifolds_drops_static_static_pair():
    """Two overlapping statics yield no manifold, so the solve never divides by zero."""
    a = Polygon.create_rectangle(2.0, 2.0, 1.0, (90, 90, 90), is_static=True)
    b = Polygon.create_rectangle(2.0, 2.0, 1.0, (90, 90, 90), is_static=True)
    a.move_to(Matrix.vector([0, 0]))
    b.move_to(Matrix.vector([1.0, 0]))
    a.physics = b.physics = False
    assert solver.build_group_manifolds([(a, b)], None) == []


def test_build_group_manifolds_keeps_dynamic_static_pair():
    """A dynamic-static overlap still builds: the guard drops only static-static."""
    dynamic = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0, 0]))
    static = Polygon.create_rectangle(2.0, 2.0, 1.0, (90, 90, 90), is_static=True)
    static.move_to(Matrix.vector([1.0, 0]))
    dynamic.physics = True
    static.physics = False
    assert len(solver.build_group_manifolds([(dynamic, static)], None)) == 1


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
    solver.solve_group_substep(engine.physics, cand_bodies, cand_pairs,
                               gravity, sub_dt, engine.num_substeps,
                               engine.num_velocity_iterations, None)

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
    solver.solve_group_substep(engine.physics, cand_bodies, cand_pairs,
                               gravity, sub_dt, engine.num_substeps,
                               engine.num_velocity_iterations, None)

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


def make_height_constraint(ya, yb):
    """Build a contactless constraint whose two bodies sit at the given heights."""
    a = Polygon.create_rectangle(2, 2, 1.0, (1, 1, 1)).move_to(Matrix.vector([0, ya]))
    b = Polygon.create_rectangle(2, 2, 1.0, (1, 1, 1)).move_to(Matrix.vector([0, yb]))
    a.physics = b.physics = True
    return Constraint(PhysicsMode.FRICTION, a, b, UP, ())


def test_constraint_height_is_mean_y():
    """The ordering key is the mean y of the two bodies."""
    constraint = make_height_constraint(4.0, 10.0)
    assert solver.constraint_height(constraint) == 7.0


def test_constraint_height_sort_is_apex_first_and_stable():
    """Ascending sort visits the apex (smallest y) first; ties keep input order."""
    low = make_height_constraint(10.0, 10.0)
    apex = make_height_constraint(2.0, 2.0)
    mid_first = make_height_constraint(5.0, 5.0)
    mid_second = make_height_constraint(5.0, 5.0)
    constraints = [low, mid_first, apex, mid_second]

    constraints.sort(key=solver.constraint_height)

    assert constraints[0] is apex
    assert constraints[1] is mid_first
    assert constraints[2] is mid_second
    assert constraints[3] is low


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
