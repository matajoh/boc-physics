"""Tests for the physics engine step driver."""

import random

from bocpy import Matrix
import pytest

from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind
from bocphysics.engine import PhysicsEngine
from bocphysics.scene import (make_golden_scene, make_pachinko_scene,
                              make_stack_scene)
from bocphysics.simulation import MAX_PHYSICS_DT


def make_engine(num_substeps=None) -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    extra = {} if num_substeps is None else {"num_substeps": num_substeps}
    return PhysicsEngine(1200, 900,
                         DetectionKind.QUADTREE, show_contacts=False, **extra)


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

    assert (floor.position.x, floor.position.y) == floor_before
    assert (wall.position.x, wall.position.y) == wall_before


def test_static_dynamic_render_partition_is_disjoint_and_complete():
    """The render cache split (render & not physics) covers statics exactly once."""
    engine = make_engine()
    for body in make_pachinko_scene().build():
        engine.add_body(body)
    ball = Circle.create(0.6, 2.0, (10, 20, 30), is_static=False)
    engine.add_body(ball.move_to(Matrix.vector([0, 0])))

    renderable = [b for b in engine.bodies if b.render]
    statics = [b for b in engine.bodies if b.render and not b.physics]
    dynamics = [b for b in engine.bodies if b.render and b.physics]

    assert set(map(id, statics)).isdisjoint(map(id, dynamics))
    assert len(statics) + len(dynamics) == len(renderable)
    assert dynamics == [ball]
    assert ball not in statics
    assert len(statics) == len(renderable) - 1


def test_remove_outside_culls_dynamics_but_keeps_statics():
    """Statics never move, so out-of-bounds culling must ignore them (cache stays valid)."""
    engine = make_engine()
    static = Circle.create(0.5, 2.0, (10, 20, 30), is_static=True)
    dynamic = Circle.create(0.5, 2.0, (10, 20, 30), is_static=False)
    far = Matrix.vector([10_000, 10_000])
    engine.add_body(static.move_to(far))
    engine.add_body(dynamic.move_to(far))

    engine.remove_outside()

    assert static in engine.bodies
    assert dynamic not in engine.bodies


def test_portrait_world_keeps_bodies_in_top_band():
    """A portrait world keeps top-band bodies a width-based top edge would clip."""
    engine = PhysicsEngine(600, 900,
                           DetectionKind.QUADTREE, show_contacts=False,
                           height_in_meters=36)
    body = Circle.create(0.6, 2.0, (10, 20, 30)).move_to(Matrix.vector([0, -15]))
    engine.add_body(body)

    engine.remove_outside()

    assert body in engine.bodies


def populate_random(engine, count: int, seed: int):
    """Drop a deterministic spread of dynamic circles into the engine."""
    rng = random.Random(seed)
    for _ in range(count):
        x = rng.uniform(-11, 11)
        y = rng.uniform(-13, 5)
        engine.add_body(Circle.create(rng.uniform(0.6, 1.2), 2.0, (200, 100, 50))
                        .move_to(Matrix.vector([x, y])))


def test_isolated_body_falls_under_gravity():
    """A lone body still integrates so the body accelerates downward."""
    engine = make_engine()
    body = Circle.create(1.0, 2.0, (200, 100, 50))
    engine.add_body(body.move_to(Matrix.vector([0, 0])))

    engine.step(1 / 60)

    assert body.position.y > 0
    assert body.linear_velocity.y > 0


def test_box_settles_on_floor_without_tunneling():
    """A box dropped onto a static floor comes to rest above it."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    box = Polygon.create_rectangle(2, 2, 2.0, (50, 120, 200))
    engine.add_body(box.move_to(Matrix.vector([0, 4])))

    for _ in range(400):
        engine.step(1 / 60)

    assert box.position.y < 9
    assert box.linear_velocity.magnitude_squared() < 1e-2


def test_resolves_overlapping_pair():
    """The solver pushes apart an overlapping pair in one frame."""
    engine = make_engine()
    a = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([-0.4, 0]))
    b = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0.4, 0]))
    a.linear_velocity = Matrix.vector([2, 0])
    b.linear_velocity = Matrix.vector([-2, 0])
    engine.add_body(a)
    engine.add_body(b)

    engine.step(1 / 60)

    relative = (b.linear_velocity - a.linear_velocity).x
    assert relative >= 0


GOLDEN_SEED = 20260608
GOLDEN_FRAMES = 500
# Golden rest pose: Jacobi XPBD static-friction solver, 16 sub-steps (converged pose).
GOLDEN_STATE = [
    (-1.9802062210, 6.8250291993, 3.3667740515),
    (8.0480855896, 6.6134373343, 5.2932780834),
    (2.8055675701, 7.9089540770, 4.7106621289),
    (-10.4994673625, 6.5800375717, 5.8501777814),
    (10.3465067894, 6.3226029520, -10.8910559967),
    (4.9469646572, 6.5663365916, 0.3009887396),
    (11.7555358067, 7.8709913414, 8.1027558868),
    (7.6756516512, 8.2138361123, 4.1924167645),
    (-12.1560751350, 6.7300122886, 1.7881012230),
    (-11.4883829542, 8.2853065285, 0.7829713460),
    (-5.8680453133, 7.7615990618, 3.8570842604),
    (-9.7587663466, 8.1321553898, 2.2017370920),
    (-3.4649045953, 8.0188807782, 4.7146781731),
    (1.0699874519, 8.2451565115, 12.5671902121),
    (6.1781895832, 8.4077027185, 0.7883724033),
    (9.2845090999, 8.1012493587, 10.9980455473),
    (-7.1847542290, 8.3659125848, 6.0024731177),
    (6.5143435108, 7.0425625795, 2.8440464799),
    (2.5789954774, 5.7341649844, 1.5670624120),
    (-0.9901047082, 8.2408533925, 5.2397016166),
    (12.1378479162, 5.9168493332, 3.9097291764),
    (-8.0957236738, 7.1855627645, 0.2800435103),
    (4.6816534009, 8.2052161987, 0.9141986917),
    (-4.8353708669, 6.7646370341, 3.0739038856),
]


def build_golden_scene(engine, seed):
    """Drop a deterministic seeded scatter of shapes onto a static floor."""
    for body in make_golden_scene(seed).build():
        engine.add_body(body)


def test_golden_master_state_is_reproducible():
    """The fixed scatter scene settles to its recorded golden state.

    Description:
        This is the determinism oracle for the engine. A fixed seed and frame
        count drive a 24-body scatter to a recorded final state. Any change that
        perturbs the physics must reproduce these values exactly.
    """
    engine = make_engine(num_substeps=16)
    build_golden_scene(engine, GOLDEN_SEED)
    for _ in range(GOLDEN_FRAMES):
        engine.step(1 / 60)

    dynamic = [body for body in engine.bodies if body.physics]
    assert len(dynamic) == len(GOLDEN_STATE)
    for body, (x, y, angle) in zip(dynamic, GOLDEN_STATE):
        assert body.position.x == pytest.approx(x, abs=1e-6)
        assert body.position.y == pytest.approx(y, abs=1e-6)
        assert body.angle.x == pytest.approx(angle, abs=1e-6)


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


def test_stack_is_stable_at_the_max_physics_step():
    """The GUI dt clamp keeps a tall stack stable; bumping it past this would explode.

    Description:
        The interactive loop caps the physics step at MAX_PHYSICS_DT so a slow
        render or startup frame cannot feed a huge dt into the explicit
        integrator. This gate pins that constant inside the stable regime: an
        eight-box stack stepped at exactly MAX_PHYSICS_DT must not gain energy.
    """
    engine = make_engine(num_substeps=8)
    for body in make_stack_scene(8).build():
        engine.add_body(body)

    peak = 0.0
    for _ in range(300):
        engine.step(MAX_PHYSICS_DT)
        peak = max(peak, max(b.linear_velocity.magnitude()
                             for b in engine.bodies if b.physics))

    assert peak < 1.0
