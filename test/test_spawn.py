"""Tests for the runtime spawn queue."""

from bocpy import Matrix

from bocphysics.bodies import Circle
from bocphysics.spawn import fit_position, spawn_overlaps, SpawnQueue


def make_circle(x, y, radius=1.0):
    """Build a unit-density dynamic circle at (x, y)."""
    return Circle.create(radius, 1.0, (200, 200, 200)).move_to(Matrix.vector([x, y]))


def test_clear_spawn_is_admitted_immediately():
    """A body queued into empty space enters on the first process pass."""
    queue = SpawnQueue()
    body = make_circle(0, 0)
    queue.enqueue(body)

    admitted = queue.process([])

    assert admitted == [body]
    assert queue.pending == []


def test_overlapping_spawn_is_nudged_into_place():
    """A body dropped onto a blocker with room is eased out and admitted."""
    resident = make_circle(0, 0)
    spawned = make_circle(0.5, 0)
    queue = SpawnQueue()
    queue.enqueue(spawned)

    admitted = queue.process([resident])

    assert admitted == [spawned]
    assert queue.pending == []
    assert not spawn_overlaps(spawned, [resident], queue.clearance)


def test_overlapping_spawn_waits_when_it_cannot_be_nudged():
    """With nudging off, an overlapping spawn is held back, not admitted."""
    resident = make_circle(0, 0)
    queue = SpawnQueue(nudge_steps=0)
    queue.enqueue(make_circle(0.5, 0))

    admitted = queue.process([resident])

    assert admitted == []
    assert len(queue.pending) == 1
    assert queue.pending[0][1] == 1


def test_spawn_is_admitted_once_space_clears():
    """A held body enters as soon as the blocker has moved away."""
    resident = make_circle(0, 0)
    spawned = make_circle(0.5, 0)
    queue = SpawnQueue(nudge_steps=0)
    queue.enqueue(spawned)

    assert queue.process([resident]) == []
    resident.move_to(Matrix.vector([10, 0]))

    assert queue.process([resident]) == [spawned]
    assert queue.pending == []


def test_spawn_is_discarded_after_the_try_budget():
    """A body that never finds space is dropped once its tries run out."""
    resident = make_circle(0, 0)
    spawned = make_circle(0.5, 0)
    queue = SpawnQueue(max_tries=3, nudge_steps=0)
    queue.enqueue(spawned)

    for _ in range(2):
        assert queue.process([resident]) == []
        assert len(queue.pending) == 1

    assert queue.process([resident]) == []
    assert queue.pending == []


def test_same_pass_spawns_do_not_overlap_each_other():
    """Two bodies queued onto the same spot admit one and hold the other."""
    queue = SpawnQueue(nudge_steps=0)
    first = make_circle(0, 0)
    second = make_circle(0.5, 0)
    queue.enqueue(first)
    queue.enqueue(second)

    admitted = queue.process([])

    assert admitted == [first]
    assert len(queue.pending) == 1
    assert queue.pending[0][0] is second


def test_fit_position_eases_a_body_off_a_blocker():
    """Nudging lifts an overlapping body to a clear pose and reports success."""
    resident = make_circle(0, 0)
    body = make_circle(0.5, 0)

    assert fit_position(body, [resident], clearance=0.05, steps=8) is True
    assert not spawn_overlaps(body, [resident], 0.05)


def test_fit_position_reports_failure_without_a_nudge_budget():
    """With no steps, an overlapping body cannot be eased clear."""
    resident = make_circle(0, 0)
    body = make_circle(0.5, 0)

    assert fit_position(body, [resident], clearance=0.05, steps=0) is False


def test_spawn_overlaps_respects_clearance():
    """A shallow overlap below the clearance does not count as overlapping."""
    # The narrow phase shrinks circle radii by 3%, so contact starts at 1.94.
    resident = make_circle(0, 0, radius=1.0)
    grazing = make_circle(1.91, 0, radius=1.0)

    assert spawn_overlaps(grazing, [resident], clearance=0.05) is False
    assert spawn_overlaps(grazing, [resident], clearance=0.01) is True
