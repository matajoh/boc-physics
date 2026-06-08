"""Tests for the physics engine step driver."""

import random

from bocpy import Matrix
import pytest

from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine


def make_engine() -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                         DetectionKind.QUADTREE, show_contacts=False)


def test_overlapping_static_bodies_do_not_crash():
    """Two overlapping static bodies must not trigger collision response."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    wall = Polygon.create_rectangle(2, 24, 2.0, (100, 100, 100), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    engine.add_body(wall.move_to(Matrix.vector([-14, -2])))

    floor_before = (floor.position.x, floor.position.y)
    wall_before = (wall.position.x, wall.position.y)

    engine.step(1 / 60)

    # static bodies are never integrated, so they stay exactly put
    assert (floor.position.x, floor.position.y) == floor_before
    assert (wall.position.x, wall.position.y) == wall_before


def populate_random(engine, count: int, seed: int):
    """Drop a deterministic spread of dynamic circles into the engine."""
    rng = random.Random(seed)
    for _ in range(count):
        x = rng.uniform(-11, 11)
        y = rng.uniform(-13, 5)
        engine.add_body(Circle.create(rng.uniform(0.6, 1.2), 2.0, (200, 100, 50))
                        .move_to(Matrix.vector([x, y])))


def test_islands_partition_dynamic_bodies_disjointly():
    """Every dynamic body belongs to exactly one island."""
    for seed in range(20):
        engine = make_engine()
        floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
        engine.add_body(floor.move_to(Matrix.vector([0, 10])))
        populate_random(engine, 25, seed)
        engine.update_swept_aabbs(1 / 60)
        engine.collisions.clear()
        engine.broad_phase()
        islands = engine.build_islands()

        seen = []
        dynamic = [b for b in engine.bodies if b.physics]
        for island in islands:
            for body in island.bodies:
                seen.append(body)

        # the islands together cover every dynamic body exactly once
        assert len(seen) == len(dynamic)
        assert len(set(map(id, seen))) == len(dynamic)
        assert set(map(id, seen)) == set(map(id, dynamic))


def test_island_pairs_have_a_dynamic_member():
    """No island holds a static-static candidate pair."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    wall = Polygon.create_rectangle(2, 24, 2.0, (100, 100, 100), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    engine.add_body(wall.move_to(Matrix.vector([-14, -2])))
    populate_random(engine, 20, seed=7)
    engine.update_swept_aabbs(1 / 60)
    engine.collisions.clear()
    engine.broad_phase()
    islands = engine.build_islands()

    for island in islands:
        for a, b in island.pairs:
            assert a.physics or b.physics


def test_isolated_body_forms_a_singleton_island():
    """A lone falling body becomes its own one-body island."""
    engine = make_engine()
    engine.add_body(Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0, 0])))
    engine.update_swept_aabbs(1 / 60)
    engine.collisions.clear()
    engine.broad_phase()
    islands = engine.build_islands()

    assert len(islands) == 1
    assert len(islands[0].bodies) == 1
    assert islands[0].pairs == []


def test_isolated_body_falls_under_gravity():
    """A singleton island still integrates so the body accelerates downward."""
    engine = make_engine()
    body = Circle.create(1.0, 2.0, (200, 100, 50))
    engine.add_body(body.move_to(Matrix.vector([0, 0])))

    engine.step(1 / 60)

    # gravity is +y in the engine's y-down world, so the body moves downward
    assert body.position.y > 0
    assert body.linear_velocity.y > 0


def make_substep_engine() -> PhysicsEngine:
    """Create a windowless engine using the substep solver."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                         DetectionKind.QUADTREE, show_contacts=False,
                         substep_solver=True)


def test_build_manifold_returns_none_for_disjoint_pair():
    """Two separated bodies have no manifold."""
    engine = make_engine()
    a = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0, 0]))
    b = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([10, 0]))
    engine.add_body(a)
    engine.add_body(b)

    assert engine.build_manifold(a, b) is None


def test_build_manifold_returns_tuple_for_overlapping_pair():
    """Two overlapping bodies yield a five-element manifold tuple."""
    engine = make_engine()
    a = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0, 0]))
    b = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([1, 0]))
    engine.add_body(a)
    engine.add_body(b)

    manifold = engine.build_manifold(a, b)
    assert manifold is not None
    assert len(manifold) == 5
    assert manifold[0] is a and manifold[1] is b


def test_substep_solver_isolated_body_falls_under_gravity():
    """A singleton island still integrates under the substep solver."""
    engine = make_substep_engine()
    body = Circle.create(1.0, 2.0, (200, 100, 50))
    engine.add_body(body.move_to(Matrix.vector([0, 0])))

    engine.step(1 / 60)

    assert body.position.y > 0
    assert body.linear_velocity.y > 0


def test_substep_solver_box_settles_on_floor_without_tunneling():
    """A box dropped onto a static floor comes to rest above it."""
    engine = make_substep_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    box = Polygon.create_rectangle(2, 2, 2.0, (50, 120, 200))
    engine.add_body(box.move_to(Matrix.vector([0, 4])))

    for _ in range(400):
        engine.step(1 / 60)

    # the floor spans y in [9, 11]; a rested box centre sits near y=8 and never
    # tunnels below the floor top, and its motion has damped to near rest
    assert box.position.y < 9
    assert box.linear_velocity.magnitude_squared() < 1e-2


def test_substep_solver_resolves_overlap_like_default_solver():
    """Both solvers push apart an overlapping pair in one frame."""
    for substep in (False, True):
        engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                               DetectionKind.QUADTREE, show_contacts=False,
                               substep_solver=substep)
        a = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([-0.4, 0]))
        b = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0.4, 0]))
        a.linear_velocity = Matrix.vector([2, 0])
        b.linear_velocity = Matrix.vector([-2, 0])
        engine.add_body(a)
        engine.add_body(b)

        engine.step(1 / 60)

        # the closing velocity is removed so the bodies stop approaching
        relative = (b.linear_velocity - a.linear_velocity).x
        assert relative >= 0


# the recorded final state of GOLDEN_SEED after GOLDEN_FRAMES sub-step solves;
# the engine is deterministic, so any divergence (e.g. a future concurrent
# solver) must reproduce these values to remain physically identical
GOLDEN_SEED = 20260608
GOLDEN_FRAMES = 120
GOLDEN_STATE = [
    (11.8232286021, 8.0814323798, 3.5007477565),
    (-10.9075518990, 4.9775228617, 7.6683336028),
    (4.9992174420, 7.9287971264, 4.7136058575),
    (3.6037605952, 6.3423742561, 4.8522651147),
    (-6.8121621211, 7.3082010859, 2.1830236123),
    (2.1039903738, 6.9660525975, -0.7018988022),
    (-10.4382449373, 8.1558326218, 3.1169127847),
    (1.0767833919, 8.3724444054, 7.0689426721),
    (-12.0283186554, 6.5654763374, 2.9757444956),
    (-8.1501012203, 4.9807898378, 5.3292400429),
    (-8.6405711116, 7.9532195885, 3.0880007044),
    (5.2149752002, 6.0709982483, 2.0249161831),
    (7.2524451749, 6.3529155053, 4.3753815139),
    (10.1486057335, 6.5000105489, 3.1881077550),
    (9.2297802735, 8.0889022020, 0.7372872588),
    (5.0840214633, 3.8637798557, 2.4305332856),
    (6.9670831175, 8.4894496631, 2.6200196438),
    (-3.5314632094, 7.9062073113, 4.7032748484),
    (3.0867795886, 8.1793019771, 1.2366297182),
    (-1.0243795282, 6.4967841211, 2.3526060226),
    (-2.1823696536, 5.2321996129, -0.2098933590),
    (0.8229077362, 5.8714845634, 3.8721586753),
    (-5.2013632931, 8.2869242170, 6.9251574256),
    (-1.3242325893, 8.1394886384, -0.0015195992),
]


def build_golden_scene(engine, seed):
    """Drop a deterministic seeded scatter of shapes onto a static floor."""
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    rng = random.Random(seed)
    for _ in range(24):
        x = rng.uniform(-12, 12)
        y = rng.uniform(-12, 6)
        angle = rng.uniform(0, 6.28)
        kind = rng.random()
        if kind < 0.4:
            body = Circle.create(rng.uniform(0.6, 1.2), 2.0, (200, 100, 50))
        elif kind < 0.7:
            body = Polygon.create_rectangle(rng.uniform(1.2, 2.2),
                                            rng.uniform(1.2, 2.2), 2.0, (50, 120, 200))
        else:
            body = Polygon.create_regular_polygon(rng.randint(3, 6),
                                                  rng.uniform(0.8, 1.3), 2.0, (180, 60, 160))

        engine.add_body(body.move_to(Matrix.vector([x, y])).rotate_to(angle))


def test_golden_master_state_is_reproducible():
    """The fixed multi-island scene settles to its recorded golden state.

    Description:
        This is the determinism oracle for the engine. A fixed seed and frame
        count drive a 24-body, multi-island scatter to a recorded final state.
        Any change that perturbs the physics, including a future concurrent
        solver that reorders island work, must reproduce these values exactly.
    """
    engine = make_engine()
    build_golden_scene(engine, GOLDEN_SEED)
    for _ in range(GOLDEN_FRAMES):
        engine.step(1 / 60)

    dynamic = [body for body in engine.bodies if body.physics]
    assert len(dynamic) == len(GOLDEN_STATE)
    for body, (x, y, angle) in zip(dynamic, GOLDEN_STATE):
        assert body.position.x == pytest.approx(x, abs=1e-6)
        assert body.position.y == pytest.approx(y, abs=1e-6)
        assert body.angle == pytest.approx(angle, abs=1e-6)
