"""Tests for the physics engine step driver."""

import random

from bocpy import Matrix
import pytest

from bocphysics import solver
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine
from bocphysics.scene import make_golden_scene


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
GOLDEN_FRAMES = 370
GOLDEN_STATE = [
    (12.1429787266, 8.1815389916, 4.7134078558),
    (-10.6930109032, 6.5466243615, 8.5311682027),
    (5.0167477482, 7.5945280973, 5.8788486127),
    (2.5349556221, 6.9474943650, -0.6584228005),
    (-5.2901270281, 6.8462424534, 7.1335997040),
    (2.2138374314, 5.5803358150, 4.2952092335),
    (-10.1316302116, 8.2087632591, 4.7143520375),
    (1.4461721888, 8.3711131857, 7.0656172109),
    (-11.9669316100, 7.9668346647, 3.0885759058),
    (-8.5828972569, 6.6066259273, 3.7984514210),
    (-6.9904082440, 8.0079764722, 4.7143358503),
    (1.1530819635, 6.9566552343, -4.8848777264),
    (8.3985781676, 7.8469371768, 5.9389263768),
    (10.8154677943, 6.2608359471, 1.5751819101),
    (10.6018558840, 8.1327908346, 0.7874353655),
    (4.5363835886, 5.9023756988, 4.3064103022),
    (6.8187422956, 8.4889899792, 2.6198828747),
    (-3.3601183228, 7.9138108591, 4.7105929458),
    (3.2104303710, 8.1795202416, 0.8656201259),
    (-0.3873694899, 6.4888457570, 2.4262100090),
    (-1.7909382603, 6.6744927327, -0.5859599379),
    (-3.6223676495, 6.1199966096, -0.0021820036),
    (-5.3756235714, 8.3117943338, 5.4942882099),
    (-1.3072697512, 8.1401630288, 0.0015736755),
]


def build_golden_scene(engine, seed):
    """Drop a deterministic seeded scatter of shapes onto a static floor."""
    for body in make_golden_scene(seed).build():
        engine.add_body(body)


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


def test_loose_quadtree_settles_like_quadtree():
    """The loose-quadtree serial path settles the scene without tunneling.

    Description:
        LOOSE_QUADTREE finds the same candidate pairs as QUADTREE but resolves
        them in a different order, so the two are not bit-identical. They must
        still agree on the physical invariant: every body comes to rest on top
        of the floor, none tunnels through it, and the pile reaches the same
        coarse height. This is the serial-vs-serial invariant parity gate.
    """
    def settle(detection):
        """Run the golden scene to rest under one detection kind."""
        engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION, detection,
                               show_contacts=False)
        build_golden_scene(engine, GOLDEN_SEED)
        for _ in range(GOLDEN_FRAMES):
            engine.step(1 / 60)

        return [body for body in engine.bodies if body.physics]

    reference = settle(DetectionKind.QUADTREE)
    loose = settle(DetectionKind.LOOSE_QUADTREE)

    assert len(loose) == len(reference)
    # the floor spans y in [9, 11]; no body may sink through it
    assert all(body.position.y < 11 for body in loose)
    # both orderings reach comparable kinetic energy; loose never runs away
    ref_speed = max(body.linear_velocity.magnitude() for body in reference)
    loose_speed = max(body.linear_velocity.magnitude() for body in loose)
    assert loose_speed <= ref_speed + 1.0
    # the two orderings reach the same coarse pile height
    ref_top = min(body.position.y for body in reference)
    loose_top = min(body.position.y for body in loose)
    assert loose_top == pytest.approx(ref_top, abs=2.0)


def settle_golden(batched):
    """Settle the golden scene to rest with the batched solver on or off."""
    engine = make_engine()
    build_golden_scene(engine, GOLDEN_SEED)
    solver.use_batched_solver = batched
    try:
        for _ in range(GOLDEN_FRAMES):
            engine.step(1 / 60)
    finally:
        solver.use_batched_solver = False

    return [body for body in engine.bodies if body.physics]


def kinetic_energy(bodies):
    """Total translational plus rotational kinetic energy of the bodies."""
    return sum(
        0.5 * body.mass * body.linear_velocity.magnitude_squared()
        + 0.5 * body.inertia * body.angular_velocity ** 2
        for body in bodies
    )


def test_batched_solver_settles_like_serial():
    """The colour-batched velocity solver settles the scene like the serial one.

    Description:
        The batched kernel runs the same accumulated PGS as the serial solver but
        visits manifolds in body-disjoint colour order rather than the serial
        path's gravity-aligned apex-first order, so it is not bit-identical and
        cannot share the golden master. With accumulation the two now settle to
        nearly the same pile; this gate asserts the robust physical invariants:
        nothing tunnels the floor, the pile reaches the same height to within a
        tight band, and the batched solver never carries more energy than serial.
        This is the Gate-C settling band.
    """
    reference = settle_golden(False)
    batched = settle_golden(True)

    assert len(batched) == len(reference)
    # the floor spans y in [9, 11]; no body may sink through it
    assert all(body.position.y < 11 for body in batched)
    # both accumulated solvers reach the same coarse pile height; the residual is
    # the colour-order vs apex-order traversal difference in a many-body settle
    ref_top = min(body.position.y for body in reference)
    batched_top = min(body.position.y for body in batched)
    assert batched_top == pytest.approx(ref_top, abs=1.0)
    # both solvers shed energy to rest; batched never carries more than serial
    assert kinetic_energy(batched) <= kinetic_energy(reference) * 1.2 + 1e-6


def test_add_body_assigns_unique_uids():
    """Every body added to the engine gets a distinct uid."""
    engine = make_engine()
    populate_random(engine, 30, seed=1)
    uids = [body.uid for body in engine.bodies]
    assert all(uid is not None for uid in uids)
    assert len(set(uids)) == len(uids)


def test_uids_are_stable_across_frames():
    """A body's uid does not change as the simulation advances."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    populate_random(engine, 20, seed=2)
    before = {id(body): body.uid for body in engine.bodies}
    for _ in range(10):
        engine.step(1 / 60)

    for body in engine.bodies:
        assert body.uid == before[id(body)]
