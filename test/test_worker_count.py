"""Worker-count independence and run-to-run reproducibility of the parallel solve.

Description:
    Unlike test_parallel, this module owns its runtime lifecycle so it can vary
    worker_count: each settle does its own start()/wait() cycle (wait() tears the
    BOC system fully down so a fresh start() may follow). The parallel pipeline is
    deterministic by construction -- the schedule is a pure function of the scene
    (patches in list order on disjoint cowns, seams in deterministic colour order),
    independent behaviours touch disjoint data so they commute, and the serial-loop
    solve does no cross-patch float reduction -- so the final blocks are bit-exact
    regardless of how many workers ran them, and identical run to run. This is the
    determinism-by-construction lock the cown graph promises.
"""

import random

from bocpy import Matrix, PinnedCown, quiesce, wait, when
import pytest

from bocphysics import parallel
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind
from bocphysics.engine import PhysicsEngine

SETTLE_FRAMES = 120
UID_BASE = 1_000_000
SEEDS = [7, 20260608, 99]


def build_settle_scene(engine, seed):
    """Drop a deterministic seeded scatter of shapes onto a static floor.

    Description:
        Kept self-contained rather than imported from test_parallel so this
        module's runtime lifecycle stays fully independent of that module's
        shared-runtime fixture.
    """
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


def settle_snapshot(seed, worker_count, num_slabs=None):
    """Settle the seeded scene under one worker count; return a uid-sorted snapshot.

    Description:
        Runs a full start()/step.../wait() cycle so the worker count is the only
        thing that changes. Snapshots every dynamic body's pose and velocity,
        sorted by uid, so two snapshots compare independent of body list order.
        num_slabs selects the partition strategy so both cuts -- the loose
        quadtree (None) and the equal-population slabs -- are held to the lock.
    """
    engine = PhysicsEngine(1200, 900,
                           DetectionKind.LOOSE_QUADTREE, show_contacts=False)
    engine.next_uid = UID_BASE
    build_settle_scene(engine, seed)
    stepper = parallel.ParallelStepper(engine, num_slabs=num_slabs)
    stepper.begin(worker_count=worker_count)
    for _ in range(SETTLE_FRAMES):
        if stepper.step():
            quiesce(30.0)

    snapshot = [(body.uid, body.position.x, body.position.y, body.angle,
                 body.linear_velocity.x, body.linear_velocity.y, body.angular_velocity)
                for body in engine.bodies if body.physics]
    snapshot.sort()
    wait()
    return snapshot


@pytest.mark.parametrize("seed", SEEDS)
def test_parallel_solve_is_worker_count_independent(seed):
    """One worker and four workers settle the same scene to the identical block.

    Description:
        The cown graph fixes the result regardless of how the scheduler fans the
        independent patch solves across workers. Bit-exact equality -- not an
        invariant tolerance -- is the right bar here: the schedule is the same
        partial order in both runs, so the only thing varying is parallelism,
        which must not change a single bit.
    """
    one = settle_snapshot(seed, 1)
    four = settle_snapshot(seed, 4)
    assert one == four


@pytest.mark.parametrize("seed", SEEDS)
def test_parallel_solve_is_reproducible_run_to_run(seed):
    """Two independent four-worker runs of the same scene produce the identical block.

    Description:
        Counters the intuition that a fanned-out solve drifts run to run: because
        the schedule is a pure function of the scene and independent behaviours
        commute on disjoint data, repeated runs are bit-exact. The parallel solve
        is a different linearization from the serial sweep (so it is not bit-equal
        to serial), but it is itself fully deterministic.
    """
    first = settle_snapshot(seed, 4)
    second = settle_snapshot(seed, 4)
    assert first == second


def test_slab_solve_is_worker_count_independent_and_reproducible():
    """The slab cut carries the same determinism lock as the quadtree cut.

    Description:
        Folds worker-count independence and run-to-run reproducibility into one
        case for the equal-population slab partition, deliberately at a single
        seed: each settle is a full start()/wait() cycle, so the module keeps its
        heavyweight-cycle budget small. One worker and four workers must agree
        bit-for-bit, and a repeat four-worker run must reproduce it exactly --
        the slab cut changes which contacts become seams but not the schedule's
        determinism, since the partition is a pure function of the scene built on
        main before any fan-out. It pins DEFAULT_SLABS, a fixed reference count,
        so both worker counts cut the identical partition; the shipped AUTO_SLABS
        default scales with the worker count and is covered separately.
    """
    one = settle_snapshot(SEEDS[0], 1, num_slabs=parallel.DEFAULT_SLABS)
    four = settle_snapshot(SEEDS[0], 4, num_slabs=parallel.DEFAULT_SLABS)
    again = settle_snapshot(SEEDS[0], 4, num_slabs=parallel.DEFAULT_SLABS)
    assert one == four
    assert four == again


def schedule_drain_probe(pinned):
    """Schedule one pinned behavior that records that it ran into the pinned list."""
    @when(pinned)
    def _mark(box):
        """Append a marker to the pinned list to prove the behavior executed."""
        box.value.append("ran")


def test_wait_drains_pending_pinned_behavior():
    """wait() pumps a scheduled pinned behavior to completion without an explicit pump.

    Description:
        on_close relies on wait() draining the final writeback rather than calling
        pump() itself; this proves that contract directly. A pinned behavior is
        scheduled and the test never pumps -- only wait() runs -- yet the behavior's
        effect is observed, so wait() drives pinned work to completion before it
        tears the runtime down.
    """
    holder = []
    pinned = PinnedCown(holder)
    schedule_drain_probe(pinned)
    wait()
    assert holder == ["ran"]
