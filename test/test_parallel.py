"""Tests for the parallel physics behaviours: intra, boundary, pinned writeback.

Description:
    Each behaviour is driven through a real BOC worker and compared to the serial
    solver core. The intra and boundary behaviours are checked bit-exact against
    a serial reference (the Matrix block crosses by pointer, so a worker round
    trip is lossless); the pinned writeback is checked by confirming it scattered
    every block onto the authoritative bodies on the main interpreter. The whole
    module shares one runtime, so uids are drawn from a single monotonic source:
    the worker shell cache is keyed by uid and assumes uids are never reused.
"""

import random

from bocpy import Cown, Matrix, notice_seed, PinnedCown, quiesce, start, wait
import pytest

from bocphysics import geometry, parallel, solver, transport
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine
from bocphysics.physics import Physics

GRAVITY = Matrix.vector([0, 9.81])
SUB_DT = (1 / 60) / 4
NUM_SUBSTEPS = 4
NUM_VEL = 10
SEEDS = list(range(12))

_next_uid = 0


def allocate_uids(count):
    """Hand out a run of globally-unique uids so the worker shell cache is safe."""
    global _next_uid
    base = _next_uid
    _next_uid += count
    return list(range(base, base + count))


@pytest.fixture(scope="module", autouse=True)
def boc_runtime():
    """Start one runtime for the module, seed the set-once config, stop at teardown."""
    start(worker_count=4)
    notice_seed(parallel.CONFIG_KEY,
                parallel.SolveConfig(Physics(PhysicsMode.FRICTION), GRAVITY,
                                     SUB_DT, NUM_VEL, False))
    yield
    wait()


class FakeEngine:
    """Minimal stand-in exposing the .bodies and .remove_outside() the writeback uses."""

    def __init__(self, bodies):
        """Hold the authoritative bodies and a counter proving the cull step ran."""
        self.bodies = bodies
        self.removed = []

    def remove_outside(self):
        """Record that the writeback reached the cull step on the main interpreter."""
        self.removed.append(len(self.bodies))


def build_dynamic(state):
    """Build one dynamic body from a fixed numeric state tuple."""
    (k, x, y, ang, vx, vy, spin, r, sides, w, h, poly_r) = state
    if k < 0.4:
        body = Circle.create(r, 2.0, (200, 100, 50))
    elif k < 0.7:
        body = Polygon.create_rectangle(w, h, 2.0, (50, 120, 200))
    else:
        body = Polygon.create_regular_polygon(sides, poly_r, 2.0, (180, 60, 160))

    body.physics = True
    body.collision = True
    body.render = True
    body.move_to(Matrix.vector([x, y])).rotate_to(ang)
    body.linear_velocity = Matrix.vector([vx, vy])
    body.angular_velocity = spin
    return body


def random_state(rng):
    """Produce one fixed numeric state tuple, clustered so pairs actually touch."""
    return (rng.random(), rng.uniform(-2, 2), rng.uniform(-2, 2),
            rng.uniform(0, 6.28), rng.uniform(-4, 4), rng.uniform(-4, 4),
            rng.uniform(-3, 3), rng.uniform(0.8, 1.4), rng.randint(3, 6),
            rng.uniform(1.4, 2.4), rng.uniform(1.4, 2.4), rng.uniform(1.0, 1.6))


def make_static_floor(uid):
    """Build a wide static floor body with the given uid."""
    floor = Polygon.create_rectangle(20.0, 1.0, 1.0, (90, 90, 90), is_static=True)
    floor.physics = False
    floor.collision = True
    floor.render = True
    floor.move_to(Matrix.vector([0, 3.0]))
    floor.uid = uid
    return floor


def build_patch_scene(seed):
    """Build a single-patch scene: clustered dynamics over one static floor.

    Description:
        Returns the dynamics, the floor, the interior uid pairs (every dynamic
        pair plus each dynamic against the floor), and the geometry snapshot. The
        uids come from the module-global source so they never collide across the
        parametrised cases that share the worker shell cache.
    """
    rng = random.Random(seed)
    dynamics = [build_dynamic(random_state(rng)) for _ in range(rng.randint(2, 5))]
    uids = allocate_uids(len(dynamics) + 1)
    for body, uid in zip(dynamics, uids):
        body.uid = uid

    floor = make_static_floor(uids[-1])
    bodies = dynamics + [floor]

    interior_uid_pairs = []
    for i in range(len(dynamics)):
        for j in range(i + 1, len(dynamics)):
            interior_uid_pairs.append((dynamics[i].uid, dynamics[j].uid))

        interior_uid_pairs.append((dynamics[i].uid, floor.uid))

    geom = geometry.build_geometry(bodies)
    return dynamics, floor, interior_uid_pairs, geom


def build_boundary_scene(seed):
    """Build a two-patch scene whose cross pairs overlap so seams produce contacts.

    Description:
        Each boundary pair is one dynamic body in patch A and one in patch B,
        placed on top of each other so the narrow phase yields a real manifold.
        Returns both patches' dynamics, the seam uid pairs in endpoint order, and
        the merged geometry snapshot. uids come from the module-global source.
    """
    rng = random.Random(seed)
    num_pairs = rng.randint(1, 4)
    dynamics_a = []
    dynamics_b = []
    boundary_uid_pairs = []
    uids = allocate_uids(2 * num_pairs)
    for index in range(num_pairs):
        cx, cy = rng.uniform(-2, 2), rng.uniform(-2, 2)
        body_a = build_dynamic(random_state(rng))
        body_b = build_dynamic(random_state(rng))
        body_a.move_to(Matrix.vector([cx, cy]))
        body_b.move_to(Matrix.vector([cx + rng.uniform(-0.3, 0.3),
                                      cy + rng.uniform(-0.3, 0.3)]))
        body_a.uid = uids[2 * index]
        body_b.uid = uids[2 * index + 1]
        dynamics_a.append(body_a)
        dynamics_b.append(body_b)
        boundary_uid_pairs.append((body_a.uid, body_b.uid))

    geom = geometry.build_geometry(dynamics_a + dynamics_b)
    return dynamics_a, dynamics_b, boundary_uid_pairs, geom


def build_writeback_dynamic(rng, uid):
    """Build one dynamic body with a known uid at the origin."""
    k = rng.random()
    if k < 0.5:
        body = Circle.create(rng.uniform(0.8, 1.4), 2.0, (200, 100, 50))
    else:
        body = Polygon.create_rectangle(1.6, 1.2, 2.0, (50, 120, 200))

    body.physics = True
    body.uid = uid
    return body


def randomise_block(rng, bodies):
    """Pack a block whose rows are these bodies' uids with fresh random state."""
    block = transport.pack_state(bodies)
    for i in range(block.rows):
        block[i, transport.POSITION.start] = rng.uniform(-5, 5)
        block[i, transport.POSITION.start + 1] = rng.uniform(-5, 5)
        block[i, transport.VELOCITY.start] = rng.uniform(-4, 4)
        block[i, transport.VELOCITY.start + 1] = rng.uniform(-4, 4)
        block[i, transport.ANGLE] = rng.uniform(0, 6.28)
        block[i, transport.SPIN] = rng.uniform(-3, 3)

    return block


def serial_intra_reference(dynamics, floor, interior_uid_pairs, batched=False):
    """Run one intra sub-step serially and return the resulting state block."""
    physics = Physics(PhysicsMode.FRICTION)
    by_uid = {body.uid: body for body in dynamics + [floor]}
    pairs = [(by_uid[ua], by_uid[ub]) for ua, ub in interior_uid_pairs]
    solver.integrate_block(dynamics, GRAVITY, SUB_DT)
    manifolds = solver.build_group_manifolds(pairs, None)
    for manifold in manifolds:
        solver.separate_manifold(manifold)
    solver.resolve_pair_list(physics, manifolds, NUM_VEL, batched)
    return transport.pack_state(dynamics)


def serial_boundary_reference(dynamics_a, dynamics_b, boundary_uid_pairs):
    """Resolve the seam serially and return both resulting state blocks."""
    physics = Physics(PhysicsMode.FRICTION)
    by_uid = {body.uid: body for body in dynamics_a + dynamics_b}
    pairs = [(by_uid[ua], by_uid[ub]) for ua, ub in boundary_uid_pairs]
    manifolds = solver.build_group_manifolds(pairs, None)
    for manifold in manifolds:
        solver.separate_manifold(manifold)
    solver.resolve_pair_list(physics, manifolds, NUM_VEL)
    return transport.pack_state(dynamics_a), transport.pack_state(dynamics_b)


def assert_blocks_equal(actual, expected):
    """Assert two packed blocks are identical to the last bit."""
    assert actual.rows == expected.rows
    assert actual.columns == expected.columns
    for i in range(actual.rows):
        for j in range(actual.columns):
            assert actual[i, j] == expected[i, j]


def assert_body_matches_row(body, block, row):
    """Assert one authoritative body equals the block row it was scattered from."""
    assert body.position.x == block[row, transport.POSITION.start]
    assert body.position.y == block[row, transport.POSITION.start + 1]
    assert body.linear_velocity.x == block[row, transport.VELOCITY.start]
    assert body.linear_velocity.y == block[row, transport.VELOCITY.start + 1]
    assert body.angle == block[row, transport.ANGLE]
    assert body.angular_velocity == block[row, transport.SPIN]


@pytest.mark.parametrize("seed", SEEDS)
def test_intra_behaviour_matches_serial(seed):
    """One intra sub-step through a worker is bit-exact with the serial core."""
    dynamics, floor, interior_uid_pairs, geom = build_patch_scene(seed)
    notice_seed(parallel.GEOMETRY_KEY, geom)
    state_cown = Cown(transport.pack_state(dynamics))
    pairs_cown = Cown(transport.pack_pairs(interior_uid_pairs))

    reference = serial_intra_reference(dynamics, floor, interior_uid_pairs)
    parallel.schedule_intra(state_cown, pairs_cown)
    quiesce(30.0)

    assert_blocks_equal(state_cown.unwrap(), reference)


@pytest.mark.parametrize("seed", SEEDS)
def test_boundary_behaviour_matches_serial(seed):
    """One boundary sub-step through a worker is bit-exact with the serial core."""
    dyn_a, dyn_b, boundary_uid_pairs, geom = build_boundary_scene(seed)
    notice_seed(parallel.GEOMETRY_KEY, geom)
    state_a = Cown(transport.pack_state(dyn_a))
    state_b = Cown(transport.pack_state(dyn_b))
    pairs_cown = Cown(transport.pack_pairs(boundary_uid_pairs))

    ref_a, ref_b = serial_boundary_reference(dyn_a, dyn_b, boundary_uid_pairs)
    parallel.schedule_boundary(state_a, state_b, pairs_cown)
    quiesce(30.0)

    assert_blocks_equal(state_a.unwrap(), ref_a)
    assert_blocks_equal(state_b.unwrap(), ref_b)


@pytest.mark.parametrize("seed", SEEDS)
def test_intra_behaviour_uses_batched_flag(seed):
    """With config.batched set, the worker runs the batched kernel, not the loop.

    Description:
        The batched flag rides the noticeboard config, so a worker honours it the
        same way the serial path honours the module global. Resolving the same
        scene serially through the batched kernel and through the worker must give
        the identical block: both call resolve_batched, so the result is exact
        regardless of colour order. Restores the shared config so later tests keep
        the serial loop.
    """
    dynamics, floor, interior_uid_pairs, geom = build_patch_scene(seed)
    notice_seed(parallel.GEOMETRY_KEY, geom)
    state_cown = Cown(transport.pack_state(dynamics))
    pairs_cown = Cown(transport.pack_pairs(interior_uid_pairs))

    reference = serial_intra_reference(dynamics, floor, interior_uid_pairs,
                                       batched=True)
    notice_seed(parallel.CONFIG_KEY,
                parallel.SolveConfig(Physics(PhysicsMode.FRICTION), GRAVITY,
                                     SUB_DT, NUM_VEL, True))
    try:
        parallel.schedule_intra(state_cown, pairs_cown)
        quiesce(30.0)
    finally:
        notice_seed(parallel.CONFIG_KEY,
                    parallel.SolveConfig(Physics(PhysicsMode.FRICTION), GRAVITY,
                                         SUB_DT, NUM_VEL, False))

    assert_blocks_equal(state_cown.unwrap(), reference)


def test_pinned_writeback_scatters_every_block_on_main():
    """The pinned writeback scatters all patch blocks onto the real bodies once."""
    rng = random.Random(0)
    uids = allocate_uids(6)
    bodies = [build_writeback_dynamic(rng, uid) for uid in uids]
    engine = FakeEngine(bodies)
    by_uid = {body.uid: body for body in bodies}

    block_a = randomise_block(rng, bodies[:4])
    block_b = randomise_block(rng, bodies[4:])
    state_cowns = [Cown(block_a), Cown(block_b)]
    engine_pinned = PinnedCown(engine)

    parallel.schedule_writeback(state_cowns, engine_pinned)
    quiesce(30.0)

    for block_cown in state_cowns:
        block = block_cown.unwrap()
        for row, uid in enumerate(transport.uids_of(block)):
            assert_body_matches_row(by_uid[uid], block, row)

    assert engine.removed == [6], "remove_outside must run exactly once on main"


def seam_colors(order, num_patches):
    """Re-derive the colour of each seam key from a colour-ordered emission list.

    Description:
        Greedily re-colours in emission order, mirroring colored_seam_order, so a
        test can assert the independent-set property without reaching into the
        helper's internals.
    """
    used = [set() for _ in range(num_patches)]
    colors = {}
    for (i, j) in sorted(order):
        c = 0
        while c in used[i] or c in used[j]:
            c += 1
        colors[(i, j)] = c
        used[i].add(c)
        used[j].add(c)
    return colors


@pytest.mark.parametrize("seed", SEEDS)
def test_colored_seam_order_is_a_permutation(seed):
    """Colour ordering reorders the seam keys without adding or dropping any."""
    rng = random.Random(seed)
    num_patches = rng.randint(2, 9)
    keys = sorted({(min(a, b), max(a, b))
                   for a, b in ((rng.randrange(num_patches), rng.randrange(num_patches))
                                for _ in range(20)) if a != b})
    order = parallel.colored_seam_order(list(keys), num_patches)
    assert sorted(order) == sorted(keys)


@pytest.mark.parametrize("seed", SEEDS)
def test_colored_seam_order_separates_patch_sharing_seams(seed):
    """Two seams sharing a patch never get the same colour (independent sets)."""
    rng = random.Random(seed)
    num_patches = rng.randint(2, 9)
    keys = sorted({(min(a, b), max(a, b))
                   for a, b in ((rng.randrange(num_patches), rng.randrange(num_patches))
                                for _ in range(20)) if a != b})
    order = parallel.colored_seam_order(list(keys), num_patches)
    colors = seam_colors(order, num_patches)

    by_color = {}
    for (i, j) in order:
        by_color.setdefault(colors[(i, j)], []).append((i, j))
    for batch in by_color.values():
        patches = [p for key in batch for p in key]
        assert len(patches) == len(set(patches))

    seen_colors = [colors[k] for k in order]
    assert seen_colors == sorted(seen_colors)


def test_colored_seam_order_is_input_order_independent():
    """The colour order is a pure function of the key set, not its input order."""
    num_patches = 6
    keys = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (3, 5)]
    forward = parallel.colored_seam_order(list(keys), num_patches)
    shuffled = parallel.colored_seam_order(list(reversed(keys)), num_patches)
    assert forward == shuffled


SETTLE_FRAMES = 120
SETTLE_SEEDS = [7, 20260608]


def build_settle_scene(engine, seed):
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


def settle_serial(seed):
    """Settle the scatter scene to rest on the serial engine."""
    engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                           DetectionKind.LOOSE_QUADTREE, show_contacts=False)
    build_settle_scene(engine, seed)
    for _ in range(SETTLE_FRAMES):
        engine.step(1 / 60)

    return [body for body in engine.bodies if body.physics]


def settle_parallel(seed):
    """Settle the same scene with the per-patch solve fanned across workers.

    Description:
        Uses ParallelStepper with no partition override, so it exercises the
        default -- the equal-population vertical-slab cut.
    """
    engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                           DetectionKind.LOOSE_QUADTREE, show_contacts=False)
    engine.next_uid = allocate_uids(64)[0]
    build_settle_scene(engine, seed)
    stepper = parallel.ParallelStepper(engine)
    stepper.begin()
    for _ in range(SETTLE_FRAMES):
        if stepper.step():
            quiesce(30.0)

    return [body for body in engine.bodies if body.physics]


@pytest.mark.parametrize("seed", SETTLE_SEEDS)
def test_parallel_settles_like_serial(seed):
    """Serial and parallel settle the same scene to the same physical invariants.

    Description:
        The parallel solve is a cown-ordered linearization, not the serial sweep,
        so it is NOT bit-identical to the serial result. It is, however,
        deterministic in its own right -- reproducible run to run and worker-count
        independent (locked by test_worker_count). It must still agree with serial
        on the physics: the same bodies survive, none tunnels the floor, the pile
        reaches the same coarse height, and the system sheds the same kinetic
        energy. This is the serial-vs-parallel invariant parity gate.
    """
    reference = settle_serial(seed)
    fanned = settle_parallel(seed)

    assert len(fanned) == len(reference)
    assert all(body.position.y < 11 for body in fanned)
    ref_speed = max(body.linear_velocity.magnitude() for body in reference)
    par_speed = max(body.linear_velocity.magnitude() for body in fanned)
    assert par_speed <= ref_speed + 1.0
    ref_top = min(body.position.y for body in reference)
    par_top = min(body.position.y for body in fanned)
    assert par_top == pytest.approx(ref_top, abs=2.0)


def settle_parallel_slabs(seed, num_slabs):
    """Settle the scatter scene with the equal-population vertical-slab cut."""
    engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                           DetectionKind.LOOSE_QUADTREE, show_contacts=False)
    engine.next_uid = allocate_uids(64)[0]
    build_settle_scene(engine, seed)
    stepper = parallel.ParallelStepper(engine, num_slabs=num_slabs)
    stepper.begin()
    for _ in range(SETTLE_FRAMES):
        if stepper.step():
            quiesce(30.0)

    return [body for body in engine.bodies if body.physics]


@pytest.mark.parametrize("num_slabs", [1, 4, 16])
@pytest.mark.parametrize("seed", SETTLE_SEEDS)
def test_slab_stepper_settles_like_serial(seed, num_slabs):
    """The slab-cut stepper settles the scene to the same physical invariants.

    Description:
        Selecting num_slabs swaps the loose-quadtree cut for equal-population
        vertical slabs but reuses the identical intra, colour-ordered seam, and
        writeback machinery, so it is another cown-ordered linearization (not
        bit-identical to serial). It must still agree with serial on the
        physics: the same bodies survive, none tunnels the floor, the pile
        reaches the same coarse height, and the system sheds the same energy.
        Swept over K=1 (one patch, no seams), a mid count, and K large enough to
        force one-body slabs (every interior pair becomes a seam) so the dense-
        seam and zero-seam schedules both run end-to-end through the solver.
    """
    reference = settle_serial(seed)
    fanned = settle_parallel_slabs(seed, num_slabs=num_slabs)

    assert len(fanned) == len(reference)
    assert all(body.position.y < 11 for body in fanned)
    ref_speed = max(body.linear_velocity.magnitude() for body in reference)
    par_speed = max(body.linear_velocity.magnitude() for body in fanned)
    assert par_speed <= ref_speed + 1.0
    ref_top = min(body.position.y for body in reference)
    par_top = min(body.position.y for body in fanned)
    assert par_top == pytest.approx(ref_top, abs=2.0)


@pytest.mark.parametrize("num_slabs", [0, -1])
def test_slab_count_below_one_is_rejected(num_slabs):
    """A non-positive num_slabs fails loudly rather than silently meaning one patch."""
    engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                           DetectionKind.LOOSE_QUADTREE, show_contacts=False)
    with pytest.raises(ValueError):
        parallel.ParallelStepper(engine, num_slabs=num_slabs)


def test_default_partition_is_slabs():
    """The default partition is the worker-scaled slab cut, never the quadtree."""
    engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                           DetectionKind.LOOSE_QUADTREE, show_contacts=False)
    assert parallel.ParallelStepper(engine).num_slabs == parallel.AUTO_SLABS
    assert parallel.resolve_slab_count(parallel.AUTO_SLABS, 4) == 10
    assert parallel.resolve_slab_count(parallel.AUTO_SLABS, 8) == 20
    assert parallel.resolve_slab_count(None, 8) is None
    assert parallel.DEFAULT_SLABS >= 1


def test_begin_resolves_and_preserves_auto_request():
    """begin() turns the AUTO_SLABS sentinel into a concrete worker-scaled int.

    Description:
        The default request stays the AUTO_SLABS string until begin(), which
        resolves it against the worker count. The original request is kept on
        _slab_request so a later begin() re-resolves from the sentinel rather
        than locking in the first concrete count -- the reason _slab_request
        exists at all.
    """
    engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                           DetectionKind.LOOSE_QUADTREE, show_contacts=False)
    stepper = parallel.ParallelStepper(engine)
    assert stepper.num_slabs == parallel.AUTO_SLABS

    stepper.begin()
    expected = parallel.resolve_slab_count(parallel.AUTO_SLABS, None)
    assert isinstance(stepper.num_slabs, int)
    assert stepper.num_slabs == expected
    assert stepper._slab_request == parallel.AUTO_SLABS

    stepper.begin()
    assert stepper.num_slabs == expected
    assert stepper._slab_request == parallel.AUTO_SLABS


def settle_parallel_quadtree(seed):
    """Settle the scatter scene with the loose-quadtree fallback (num_slabs=None)."""
    engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                           DetectionKind.LOOSE_QUADTREE, show_contacts=False)
    engine.next_uid = allocate_uids(64)[0]
    build_settle_scene(engine, seed)
    stepper = parallel.ParallelStepper(engine, num_slabs=None)
    stepper.begin()
    for _ in range(SETTLE_FRAMES):
        if stepper.step():
            quiesce(30.0)

    return [body for body in engine.bodies if body.physics]


@pytest.mark.parametrize("seed", SETTLE_SEEDS)
def test_quadtree_fallback_settles_like_serial(seed):
    """The retained loose-quadtree fallback still settles to the serial invariants.

    Description:
        Slabs are the default, but num_slabs=None keeps the loose-quadtree
        cut reachable; this guards that fallback against rot. Same
        invariant-parity checks as the default and slab settle tests: bodies
        survive, none tunnels the floor, the pile reaches the same coarse height,
        and the system sheds the same kinetic energy.
    """
    reference = settle_serial(seed)
    fanned = settle_parallel_quadtree(seed)

    assert len(fanned) == len(reference)
    assert all(body.position.y < 11 for body in fanned)
    ref_speed = max(body.linear_velocity.magnitude() for body in reference)
    par_speed = max(body.linear_velocity.magnitude() for body in fanned)
    assert par_speed <= ref_speed + 1.0
    ref_top = min(body.position.y for body in reference)
    par_top = min(body.position.y for body in fanned)
    assert par_top == pytest.approx(ref_top, abs=2.0)


def build_two_patch_scene(seed):
    """Build two clustered patches sharing a floor, linked by one seam pair.

    Description:
        Each patch is a small cluster of dynamics over one shared static floor;
        its interior pairs are every dynamic-dynamic pair plus each dynamic
        against the floor. A single seam pair links the first body of each patch.
        Returns both patches' dynamics, the floor, the interior uid pairs for
        each patch, the seam uid pairs in endpoint order, and the merged geometry
        snapshot. uids come from the module-global source so they never collide
        across the parametrised cases that share the worker shell cache.
    """
    rng = random.Random(seed)
    count_a = rng.randint(2, 3)
    count_b = rng.randint(2, 3)
    dyn_a = [build_dynamic(random_state(rng)) for _ in range(count_a)]
    dyn_b = [build_dynamic(random_state(rng)) for _ in range(count_b)]
    uids = allocate_uids(count_a + count_b + 1)
    for body, uid in zip(dyn_a + dyn_b, uids):
        body.uid = uid

    floor = make_static_floor(uids[-1])
    interior_a = interior_pairs_with_floor(dyn_a, floor)
    interior_b = interior_pairs_with_floor(dyn_b, floor)
    seam = [(dyn_a[0].uid, dyn_b[0].uid)]
    geom = geometry.build_geometry(dyn_a + dyn_b + [floor])
    return dyn_a, dyn_b, floor, interior_a, interior_b, seam, geom


def interior_pairs_with_floor(dynamics, floor):
    """Every dynamic-dynamic pair in a patch plus each dynamic against the floor."""
    pairs = []
    for i in range(len(dynamics)):
        for j in range(i + 1, len(dynamics)):
            pairs.append((dynamics[i].uid, dynamics[j].uid))

        pairs.append((dynamics[i].uid, floor.uid))

    return pairs


def serial_intra_step(physics, dynamics, pairs):
    """Mirror solve_intra_substep serially: integrate then resolve interior pairs."""
    solver.integrate_block(dynamics, GRAVITY, SUB_DT)
    manifolds = solver.build_group_manifolds(pairs, None)
    for manifold in manifolds:
        solver.separate_manifold(manifold)
    solver.resolve_pair_list(physics, manifolds, NUM_VEL, False)


def serial_seam_step(physics, pairs):
    """Mirror solve_boundary_substep serially: resolve the seam, no integration."""
    manifolds = solver.build_group_manifolds(pairs, None)
    for manifold in manifolds:
        solver.separate_manifold(manifold)
    solver.resolve_pair_list(physics, manifolds, NUM_VEL, False)


def serial_two_patch_reference(dyn_a, dyn_b, floor, interior_a, interior_b, seam):
    """Run the whole multi-sub-step frame serially in the worker FIFO order.

    Description:
        Reproduces the order per-cown FIFO imposes on the fan-out: for each
        sub-step, both patches integrate and resolve their interiors, then the
        seam resolves on top of both patches' fresh state. Returns the final
        packed block for each patch to compare bit-exact against the workers.
    """
    physics = Physics(PhysicsMode.FRICTION)
    by_uid = {body.uid: body for body in dyn_a + dyn_b + [floor]}
    pairs_a = [(by_uid[ua], by_uid[ub]) for ua, ub in interior_a]
    pairs_b = [(by_uid[ua], by_uid[ub]) for ua, ub in interior_b]
    pairs_seam = [(by_uid[ua], by_uid[ub]) for ua, ub in seam]
    for _ in range(NUM_SUBSTEPS):
        serial_intra_step(physics, dyn_a, pairs_a)
        serial_intra_step(physics, dyn_b, pairs_b)
        serial_seam_step(physics, pairs_seam)

    return transport.pack_state(dyn_a), transport.pack_state(dyn_b)


@pytest.mark.parametrize("seed", SEEDS)
def test_multistep_two_patch_matches_serial(seed):
    """A full multi-sub-step two-patch frame is bit-exact with the serial core.

    Description:
        This is the load-bearing FIFO-interleaving lock. The single-sub-step
        tests above check one intra or one boundary in isolation; here all
        NUM_SUBSTEPS sub-steps are scheduled up front exactly the way
        ParallelStepper.step does (every patch's intra, then the seam, per
        sub-step). Per-cown FIFO must thread them so the workers reproduce the
        serial integrate -> interior -> seam sequence to the last bit, on every
        patch cown, across all sub-steps. Because the schedule order alone
        determines the result, this also pins worker-count independence: the
        same fixed order yields the same block no matter how many workers run it.
    """
    dyn_a, dyn_b, floor, interior_a, interior_b, seam, geom = build_two_patch_scene(seed)
    notice_seed(parallel.GEOMETRY_KEY, geom)
    state_a = Cown(transport.pack_state(dyn_a))
    state_b = Cown(transport.pack_state(dyn_b))
    pairs_a = Cown(transport.pack_pairs(interior_a))
    pairs_b = Cown(transport.pack_pairs(interior_b))
    seam_cown = Cown(transport.pack_pairs(seam))

    ref_a, ref_b = serial_two_patch_reference(dyn_a, dyn_b, floor,
                                              interior_a, interior_b, seam)

    for _ in range(NUM_SUBSTEPS):
        parallel.schedule_intra(state_a, pairs_a)
        parallel.schedule_intra(state_b, pairs_b)
        parallel.schedule_boundary(state_a, state_b, seam_cown)
    quiesce(30.0)

    assert_blocks_equal(state_a.unwrap(), ref_a)
    assert_blocks_equal(state_b.unwrap(), ref_b)


def build_seam_drop_scene(drop_speed):
    """A dynamic body falling onto a left (interior) and a right (seam) support.

    Description:
        X straddles a seam: it shares an interior contact with the left support
        and a cross-seam contact with the right one. Both supports are static and
        overlap X so the narrow phase yields a real manifold for each. drop_speed
        is the downward (y-down) closing speed; tune it across restitution_threshold.
    """
    x = Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0.0, 0.0]))
    left = Circle.create(1.0, 2.0, (90, 90, 90)).move_to(Matrix.vector([-1.0, 1.5]))
    right = Circle.create(1.0, 2.0, (90, 90, 90)).move_to(Matrix.vector([1.0, 1.5]))
    x.physics = True
    left.physics = right.physics = False
    x.linear_velocity = Matrix.vector([0.0, drop_speed])
    return x, left, right


def seam_separation_speed(x, support):
    """Post-solve separating speed of X from a support along their contact normal.

    Description:
        Positive means the bodies are flying apart (a restitution bounce was
        applied); near zero means the seam merely arrested the closing motion.
    """
    collision = solver.detect_collision(x, support)
    return (-x.linear_velocity).vecdot(collision.normal)


def seam_outcomes(physics, drop_speed, iters=NUM_VEL):
    """Return (monolithic, decomposed) seam separation speeds for one drop.

    Description:
        Monolithic prepares both contacts at the start velocity and solves them
        together (the serial engine's order). Decomposed solves the interior
        contact first, then prepares and solves the seam -- the parallel path,
        where the seam's restitution target is sampled after the interior solve
        has already damped X.
    """
    x, left, right = build_seam_drop_scene(drop_speed)
    monolithic = solver.build_group_manifolds([(x, left), (x, right)], None)
    solver.resolve_pair_list(physics, monolithic, iters)
    mono = seam_separation_speed(x, right)

    x, left, right = build_seam_drop_scene(drop_speed)
    intra = solver.build_group_manifolds([(x, left)], None)
    solver.resolve_pair_list(physics, intra, iters)
    seam = solver.build_group_manifolds([(x, right)], None)
    solver.resolve_pair_list(physics, seam, iters)
    deco = seam_separation_speed(x, right)

    return mono, deco


def test_seam_decomposition_suppresses_restitution_above_threshold():
    """The parallel seam order strips the restitution bounce a serial solve keeps.

    Description:
        Characterisation lock, not an endorsement. For a body
        crossing a seam above the restitution threshold, the monolithic serial
        order applies a real bounce while the decomposed (intra-then-seam) order
        samples the restitution target after the interior solve has damped the
        body below the threshold, so the seam applies almost none. This pins the
        measured divergence so any future change to the seam ordering is caught.
    """
    physics = Physics(PhysicsMode.FRICTION, restitution=0.5, restitution_threshold=1.0)

    mono, deco = seam_outcomes(physics, drop_speed=4.0)
    assert mono > 1.0, "serial order should apply a restitution bounce at the seam"
    assert abs(deco) < 0.1, "decomposed order suppresses the seam restitution"
    assert mono - deco > 1.0, "the cross-seam restitution gap is large, not negligible"


def test_seam_decomposition_matches_serial_below_threshold():
    """Below the restitution threshold both orders agree: no bounce to diverge on.

    Description:
        Restitution is gated to zero for closing speeds at or below the
        threshold, so the seam-ordering hazard simply does not fire there. This
        is the other half of the characterisation: the divergence is confined to
        genuine above-threshold impacts, which is why resting stacks are immune.
        The accumulated PGS solver leaves a slightly larger but still negligible
        seam difference here (order 1e-3 m/s) than the per-iteration solver did.
    """
    physics = Physics(PhysicsMode.FRICTION, restitution=0.5, restitution_threshold=1.0)

    mono, deco = seam_outcomes(physics, drop_speed=0.5)
    assert abs(mono - deco) < 2e-3, "below threshold the seam order barely matters"
