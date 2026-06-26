"""Tests for the physics engine step driver."""

import random

from bocpy import Matrix
import pytest

from bocphysics import solver
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine
from bocphysics.scene import make_golden_scene, make_pachinko_scene


def make_engine(num_substeps=None) -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    extra = {} if num_substeps is None else {"num_substeps": num_substeps}
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
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
    engine = PhysicsEngine(600, 900, PhysicsMode.FRICTION,
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
GOLDEN_STATE = [
    (-0.1662585848, 6.6238184381, 1.8894089592),
    (6.5850606132, 6.4360728221, 5.7395981112),
    (1.6450787774, 8.0488908481, 3.1401531622),
    (-10.5502261540, 6.8559700528, 6.1274534748),
    (9.1706131927, 8.0355673879, 0.1883003253),
    (3.6677495788, 6.5734680544, -0.2442390870),
    (11.8709135982, 7.8709113103, 6.4092013978),
    (5.9550900541, 8.2138092252, 2.0908410451),
    (-12.1560850247, 6.7283998342, 4.4582015844),
    (-12.2560289144, 8.2852345158, -2.3539120363),
    (-6.3190991918, 7.8993822066, 4.7099987518),
    (-9.1390886551, 8.1320869058, 0.9399035000),
    (-3.6242454846, 8.0188278244, 3.4580121152),
    (-1.8351881337, 7.2070962678, 4.3344847479),
    (5.2465029567, 6.8854807826, 0.5360636780),
    (7.5341583756, 8.1012633196, 1.5684299725),
    (-7.6379174712, 8.3657609575, 3.1659534946),
    (8.4307827626, 6.4607377816, 9.0285537882),
    (1.7710530973, 6.0139616348, 1.5681800763),
    (-0.3188076107, 8.2410430236, 4.1928668192),
    (10.3229752333, 6.6183783715, -0.6317722962),
    (-6.1781917112, 5.7417018017, 2.8598030000),
    (3.9361876622, 8.0661508840, -1.1896756177),
    (-4.7753274443, 6.7236722396, 3.3034606228),
]


def build_golden_scene(engine, seed):
    """Drop a deterministic seeded scatter of shapes onto a static floor."""
    for body in make_golden_scene(seed).build():
        engine.add_body(body)


def test_golden_master_state_is_reproducible():
    """The fixed scatter scene settles to its recorded golden state.

    Description:
        This is the determinism oracle for the engine. A fixed seed and frame
        count drive a 24-body scatter to a recorded final state.
        Any change that perturbs the physics, including a future concurrent
        solver that reorders contact work, must reproduce these values exactly.
    """
    engine = make_engine(num_substeps=8)
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
    assert all(body.position.y < 11 for body in loose)
    ref_speed = max(body.linear_velocity.magnitude() for body in reference)
    loose_speed = max(body.linear_velocity.magnitude() for body in loose)
    assert loose_speed <= ref_speed + 1.0
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


@pytest.mark.xfail(reason="cross-solver window: re-unified at S3.5")
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
        This is the settling-band parity gate.
    """
    reference = settle_golden(False)
    batched = settle_golden(True)

    assert len(batched) == len(reference)
    assert all(body.position.y < 11 for body in batched)
    ref_top = min(body.position.y for body in reference)
    batched_top = min(body.position.y for body in batched)
    assert batched_top == pytest.approx(ref_top, abs=1.0)
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
