"""Tests for the loose-quadtree spatial partition."""

import random

from bocpy import Matrix
import pytest

from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine
from bocphysics.patches import build_partition, build_slab_partition, slab_boundaries
from bocphysics.quadtree import QuadTree


def make_engine(detection=DetectionKind.QUADTREE) -> PhysicsEngine:
    """Create a windowless engine with the given broad-phase detection kind."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION, detection, show_contacts=False)


def populate_cluster(engine, count: int, seed: int):
    """Drop a dense overlapping cluster of dynamic circles around the origin.

    Description:
        The cluster straddles the root cell seams at x=0 and y=0, so the
        partition reliably produces both interior and boundary pairs.
    """
    rng = random.Random(seed)
    for _ in range(count):
        x = rng.uniform(-3, 3)
        y = rng.uniform(-3, 3)
        engine.add_body(Circle.create(rng.uniform(0.6, 1.2), 2.0, (200, 100, 50))
                        .move_to(Matrix.vector([x, y])))


def candidate_pairs(engine, dt: float = 1 / 60):
    """Run the broad phase and return its raw candidate pair list."""
    engine.update_swept_aabbs(dt)
    engine.collisions.clear()
    engine.broad_phase()
    return list(engine.collisions)


def is_dynamic_pair(pair) -> bool:
    """A pair matters to the partition if at least one endpoint is dynamic."""
    a, b = pair
    return a.physics or b.physics


def populate_mixed(engine, seed: int):
    """Drop overlapping dynamics and statics so all three pair categories occur.

    Description:
        Twelve dynamic circles cluster around the origin (dynamic-dynamic and
        dynamic-static pairs), and two deliberately overlapping static boxes
        guarantee at least one static-static candidate every seed, so the test
        always exercises the partition's static-static drop.
    """
    rng = random.Random(seed)
    for _ in range(12):
        engine.add_body(Circle.create(rng.uniform(0.6, 1.2), 2.0, (200, 100, 50))
                        .move_to(Matrix.vector([rng.uniform(-3, 3), rng.uniform(-3, 3)])))

    engine.add_body(Polygon.create_rectangle(3.0, 3.0, 2.0, (90, 90, 90), is_static=True)
                    .move_to(Matrix.vector([0.0, 0.0])))
    engine.add_body(Polygon.create_rectangle(3.0, 3.0, 2.0, (90, 90, 90), is_static=True)
                    .move_to(Matrix.vector([1.0, 1.0])))


@pytest.mark.parametrize("seed", range(30))
def test_every_dynamic_body_in_exactly_one_patch(seed):
    """The patches tile the dynamic body set with no gaps or overlaps."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_partition(engine.bodies, candidates, engine.detection.box)

    dynamic = [b for b in engine.bodies if b.physics]
    seen = [body for patch in partition.patches for body in patch.bodies]
    assert len(seen) == len(dynamic)
    assert set(map(id, seen)) == set(map(id, dynamic))


@pytest.mark.parametrize("seed", range(30))
def test_every_candidate_pair_classified_exactly_once(seed):
    """No dynamic-involving candidate pair is dropped or double-counted."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_partition(engine.bodies, candidates, engine.detection.box)

    classified = [pair for patch in partition.patches for pair in patch.interior_pairs]
    classified += [boundary.pair for boundary in partition.boundary_pairs]
    expected = [pair for pair in candidates if is_dynamic_pair(pair)]

    assert len(classified) == len(expected)
    assert sorted(map(id, (p for pair in classified for p in pair))) == \
        sorted(map(id, (p for pair in expected for p in pair)))


@pytest.mark.parametrize("seed", range(30))
def test_boundary_pair_spans_its_two_patches(seed):
    """Each boundary pair records exactly the two patches its endpoints live in."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_partition(engine.bodies, candidates, engine.detection.box)

    patch_of = {}
    for index, patch in enumerate(partition.patches):
        for body in patch.bodies:
            patch_of[body.uid] = index

    for boundary in partition.boundary_pairs:
        a, b = boundary.pair
        assert {boundary.patch_a, boundary.patch_b} == {patch_of[a.uid], patch_of[b.uid]}


def test_cluster_actually_produces_boundary_pairs():
    """The dense cluster exercises the boundary path across the fuzz seeds."""
    total = 0
    for seed in range(30):
        engine = make_engine()
        populate_cluster(engine, 30, seed)
        candidates = candidate_pairs(engine)
        partition = build_partition(engine.bodies, candidates, engine.detection.box)
        total += len(partition.boundary_pairs)

    assert total > 0


def test_dynamic_static_contact_is_interior_to_dynamic_patch():
    """A dynamic body resting on a static floor makes that contact interior to its patch."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 0])))
    ball = Circle.create(1.0, 2.0, (200, 100, 50))
    engine.add_body(ball.move_to(Matrix.vector([0, -1.0])))

    candidates = candidate_pairs(engine)
    assert any(not p[0].physics or not p[1].physics for p in candidates)

    partition = build_partition(engine.bodies, candidates, engine.detection.box)
    patch_of = {b.uid: i for i, p in enumerate(partition.patches) for b in p.bodies}
    index = patch_of[ball.uid]

    assert any((floor in pair) for pair in partition.patches[index].interior_pairs)


def test_ownership_is_stable_across_rebuilds():
    """Rebuilding the partition with stable uids yields identical ownership."""
    engine = make_engine()
    populate_cluster(engine, 30, seed=11)
    candidates = candidate_pairs(engine)

    first = build_partition(engine.bodies, candidates, engine.detection.box)
    second = build_partition(engine.bodies, candidates, engine.detection.box)

    def signature(partition):
        """Reduce a partition's boundary pairs to a comparable signature."""
        return [(b.patch_a, b.patch_b, b.pair[0].uid, b.pair[1].uid)
                for b in partition.boundary_pairs]

    assert signature(first) == signature(second)


@pytest.mark.parametrize("seed", range(30))
def test_loose_detection_matches_quadtree_dynamic_pairs(seed):
    """LOOSE_QUADTREE finds the same dynamic-involving pairs as plain QUADTREE."""
    def pairs_for(detection):
        """Run the broad phase under one detection kind and key pairs by uid."""
        engine = make_engine(detection)
        populate_cluster(engine, 30, seed)
        candidates = candidate_pairs(engine)
        return {(a.uid, b.uid) for a, b in candidates if a.physics or b.physics}

    assert pairs_for(DetectionKind.LOOSE_QUADTREE) == pairs_for(DetectionKind.QUADTREE)


@pytest.mark.parametrize("seed", range(30))
def test_loose_detection_equals_quadtree_minus_static_static(seed):
    """LOOSE emits exactly the quadtree pairs minus static-static, once each.

    Description:
        On a scene mixing dynamic-dynamic, dynamic-static, and static-static
        overlaps, the loose partition must reproduce every dynamic-involving pair
        the plain quadtree finds, drop every static-static pair (it resolves to a
        no-op the solver must never see), and emit no pair twice. Pairs are keyed
        as unordered uid sets so a flipped endpoint order or normal direction does
        not register as a difference.
    """
    def pairs_for(detection):
        """Run the broad phase under one detection kind; return its raw pair list."""
        engine = make_engine(detection)
        populate_mixed(engine, seed)
        return candidate_pairs(engine)

    quad = pairs_for(DetectionKind.QUADTREE)
    loose = pairs_for(DetectionKind.LOOSE_QUADTREE)

    quad_keys = {frozenset((a.uid, b.uid)) for a, b in quad}
    static_static = {frozenset((a.uid, b.uid)) for a, b in quad
                     if not a.physics and not b.physics}
    loose_keys = [frozenset((a.uid, b.uid)) for a, b in loose]

    assert static_static, "scene produced no static-static pair to drop"
    assert len(loose_keys) == len(set(loose_keys)), "loose emitted a duplicate pair"
    assert set(loose_keys) == quad_keys - static_static


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("num_slabs", [2, 5, 8])
def test_slab_patches_are_equal_population(seed, num_slabs):
    """Equal-population slabs balance the dynamic body count within one body."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, num_slabs)

    dynamic = [b for b in engine.bodies if b.physics]
    expected_count = min(num_slabs, len(dynamic))
    sizes = [len(patch.bodies) for patch in partition.patches]
    assert len(sizes) == expected_count
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("seed", range(30))
def test_slab_tiles_dynamic_bodies_left_to_right(seed):
    """Each dynamic body lands in the equal-population slab fixed by its x-rank."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, 5)

    dynamic = sorted((b for b in engine.bodies if b.physics), key=lambda b: b.position.x)
    seen = [body for patch in partition.patches for body in patch.bodies]
    assert set(map(id, seen)) == set(map(id, dynamic))

    count = len(partition.patches)
    total = len(dynamic)
    patch_of = {b.uid: i for i, p in enumerate(partition.patches) for b in p.bodies}
    for rank, body in enumerate(dynamic):
        assert patch_of[body.uid] == min(count - 1, rank * count // total)


@pytest.mark.parametrize("seed", range(30))
def test_slab_min_population_caps_slab_count(seed):
    """A population floor collapses a small scene to fuller slabs, never one-body."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, 12, min_slab_bodies=6)

    dynamic = [b for b in engine.bodies if b.physics]
    assert len(partition.patches) == min(12, len(dynamic) // 6)
    assert all(len(patch.bodies) >= 6 for patch in partition.patches)


@pytest.mark.parametrize("seed", range(30))
def test_slab_single_patch_has_no_seams(seed):
    """One slab owns every dynamic body and every dynamic pair, with no boundary."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, 1)

    assert len(partition.patches) == 1
    assert partition.boundary_pairs == []
    dynamic = [b for b in engine.bodies if b.physics]
    assert len(partition.patches[0].bodies) == len(dynamic)
    interior = partition.patches[0].interior_pairs
    expected = [pair for pair in candidates if is_dynamic_pair(pair)]
    assert len(interior) == len(expected)


@pytest.mark.parametrize("seed", range(30))
def test_slab_every_candidate_pair_classified_once(seed):
    """No dynamic-involving candidate pair is dropped or double-counted by slabs."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, 5)

    classified = [pair for patch in partition.patches for pair in patch.interior_pairs]
    classified += [boundary.pair for boundary in partition.boundary_pairs]
    expected = [pair for pair in candidates if is_dynamic_pair(pair)]
    assert sorted(map(id, (p for pair in classified for p in pair))) == \
        sorted(map(id, (p for pair in expected for p in pair)))


@pytest.mark.parametrize("seed", range(30))
def test_slab_boundary_pair_spans_its_two_patches(seed):
    """Each slab boundary pair records exactly the two slabs its endpoints live in."""
    engine = make_engine()
    populate_cluster(engine, 30, seed)
    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, 5)

    patch_of = {}
    for index, patch in enumerate(partition.patches):
        for body in patch.bodies:
            patch_of[body.uid] = index

    for boundary in partition.boundary_pairs:
        a, b = boundary.pair
        assert {boundary.patch_a, boundary.patch_b} == {patch_of[a.uid], patch_of[b.uid]}


def test_slab_count_capped_at_body_count():
    """Asking for more slabs than bodies yields one patch per body, no empties."""
    engine = make_engine()
    populate_cluster(engine, 4, seed=3)
    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, 16)

    dynamic = [b for b in engine.bodies if b.physics]
    assert len(partition.patches) == len(dynamic)
    assert all(len(patch.bodies) == 1 for patch in partition.patches)


def test_slab_empty_world_is_empty_partition():
    """With no dynamic bodies the slab partition has no patches to schedule."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 0])))
    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, 8)

    assert partition.patches == []
    assert partition.boundary_pairs == []


def test_slab_dynamic_static_contact_is_interior_to_dynamic_patch():
    """A dynamic body on a static floor keeps that contact interior to its slab."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 0])))
    ball = Circle.create(1.0, 2.0, (200, 100, 50))
    engine.add_body(ball.move_to(Matrix.vector([0, -1.0])))

    candidates = candidate_pairs(engine)
    partition = build_slab_partition(engine.bodies, candidates,
                                     engine.detection.box, 4)
    patch_of = {b.uid: i for i, p in enumerate(partition.patches) for b in p.bodies}
    index = patch_of[ball.uid]
    assert any((floor in pair) for pair in partition.patches[index].interior_pairs)


def make_row(engine, xs):
    """Add one dynamic circle at each x along the y=0 axis."""
    for x in xs:
        engine.add_body(Circle.create(0.5, 2.0, (10, 10, 10))
                        .move_to(Matrix.vector([x, 0.0])))


def test_slab_boundaries_split_between_adjacent_bodies():
    engine = make_engine()
    make_row(engine, [-5, -3, 1, 4, 7, 9])
    partition = build_slab_partition(engine.bodies, [], engine.detection.box,
                                     num_slabs=3, min_slab_bodies=1)
    bounds = slab_boundaries(partition)
    populated = [patch for patch in partition.patches if patch.bodies]
    assert len(bounds) == len(populated) - 1
    assert bounds == sorted(bounds)
    xs = sorted(body.position.x for body in engine.bodies)
    for x in bounds:
        assert xs[0] < x < xs[-1]


def test_slab_boundaries_empty_without_dynamics():
    engine = make_engine()
    partition = build_slab_partition(engine.bodies, [], engine.detection.box, num_slabs=4)
    assert slab_boundaries(partition) == []


def test_slab_boundaries_single_slab_has_no_seam():
    engine = make_engine()
    make_row(engine, [-2, 0, 2])
    partition = build_slab_partition(engine.bodies, [], engine.detection.box, num_slabs=1)
    assert slab_boundaries(partition) == []


@pytest.mark.parametrize("seed", range(20))
def test_slab_boundaries_are_sorted_and_within_range(seed):
    engine = make_engine()
    populate_cluster(engine, 24, seed)
    partition = build_slab_partition(engine.bodies, [], engine.detection.box,
                                     num_slabs=4, min_slab_bodies=1)
    bounds = slab_boundaries(partition)
    assert bounds == sorted(bounds)
    xs = sorted(body.position.x for body in engine.bodies)
    for x in bounds:
        assert xs[0] <= x <= xs[-1]


def test_quadtree_boxes_root_only_when_empty():
    box = make_engine().detection.box
    tree = QuadTree(box)
    assert tree.boxes() == [tree.root.box]


def test_quadtree_boxes_subdivides_under_a_dense_cluster():
    engine = make_engine()
    populate_cluster(engine, 60, seed=3)
    engine.update_swept_aabbs(1 / 60)
    tree = QuadTree(engine.detection.box)
    for body in engine.bodies:
        tree.add(body)
    boxes = tree.boxes()
    assert boxes[0] is tree.root.box
    assert len(boxes) > 1
