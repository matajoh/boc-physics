"""Tests for stable feature-pair contact IDs in the narrow phase.

These guard the warm-start identity substrate: every contact point carries a
(source_uid, vertex_index) feature ID that names the incident vertex it came
from, the IDs never perturb the geometry, and they stay stable across the
frames of a settling pile.
"""

import random

from bocpy import Matrix
import pytest

from bocphysics.bodies import Circle, Polygon
from bocphysics.collisions import detect_collision
from bocphysics.config import DetectionKind
from bocphysics.contacts import (build_contacts, contact_velocity,
                                 find_contact_points,
                                 find_contact_points_polygon_polygon,
                                 relative_normal_velocity,
                                 scan_edge_points)
from bocphysics.engine import PhysicsEngine
from bocphysics.geometry import broad_box, GeometryPool

FRAME = 1 / 60


def make_circle(x, y, vx=0.0, vy=0.0, omega=0.0, radius=1.0):
    """Build a dynamic circle at (x, y) with the given motion state."""
    body = Circle.create(radius, 2.0, (200, 100, 50))
    body.physics = True
    body.move_to(Matrix.vector([x, y]))
    body.linear_velocity = Matrix.vector([vx, vy])
    body.angular_velocity.x = omega
    return body


def make_static_box(x, y, width=40.0, height=2.0):
    """Build a static rectangle floor centred at (x, y)."""
    floor = Polygon.create_rectangle(width, height, 1.0, (90, 90, 90), is_static=True)
    floor.move_to(Matrix.vector([x, y]))
    floor.physics = False
    return floor


def pool_for(*bodies) -> GeometryPool:
    """Build a GeometryPool over the polygon members of bodies for contact gen."""
    return GeometryPool([b for b in bodies if isinstance(b, Polygon)])


def make_engine() -> PhysicsEngine:
    """Create a windowless substep engine with friction physics."""
    return PhysicsEngine(1200, 900,
                         DetectionKind.QUADTREE, show_contacts=False)


def overlapping_box_pair(rng: random.Random):
    """Build two overlapping rectangles with stamped uids, or None if disjoint."""
    a = Polygon.create_rectangle(rng.uniform(2, 5), rng.uniform(2, 5),
                                 2.0, (200, 100, 50))
    b = Polygon.create_rectangle(rng.uniform(2, 5), rng.uniform(2, 5),
                                 2.0, (50, 100, 200))
    a.uid, b.uid = 11, 22
    a.move_to(Matrix.vector([0, 0]))
    b.move_to(Matrix.vector([rng.uniform(-2, 2), rng.uniform(-2, 2)]))
    a.angle.x, b.angle.x = rng.uniform(0, 0.4), rng.uniform(0, 0.4)
    a.update_needed_ = b.update_needed_ = True
    collision = detect_collision(a, b)
    if collision is None:
        return None
    return a, b, collision


def test_feature_id_names_the_incident_vertex():
    """Each contact ID (uid, idx) names the exact polygon vertex it came from."""
    rng = random.Random(20260616)
    checked = 0
    for _ in range(400):
        pair = overlapping_box_pair(rng)
        if pair is None:
            continue
        a, b, collision = pair
        c0, c1, id0, id1 = find_contact_points(a, b, collision)
        by_uid = {a.uid: a, b.uid: b}
        for point, fid in ((c0, id0), (c1, id1)):
            if point is None:
                continue
            uid, idx = fid
            source = by_uid[uid]
            vertex = source.transformed_vertices[idx]
            assert point.x == vertex.x and point.y == vertex.y
            checked += 1
    assert checked > 0


def test_feature_id_is_well_formed():
    """IDs reference a pair member, an in-range vertex, and stay distinct."""
    rng = random.Random(99)
    for _ in range(400):
        pair = overlapping_box_pair(rng)
        if pair is None:
            continue
        a, b, collision = pair
        _c0, _c1, id0, id1 = find_contact_points(a, b, collision)
        ids = [fid for fid in (id0, id1) if fid is not None]
        for uid, idx in ids:
            assert uid in (a.uid, b.uid)
            source = a if uid == a.uid else b
            assert 0 <= idx < source.transformed_vertices.rows
        if len(ids) == 2:
            assert ids[0] != ids[1]


def test_circle_contact_carries_its_own_feature_id():
    """A circle contributes a single contact with ID (circle_uid, 0)."""
    circle = Circle.create(1.0, 2.0, (200, 100, 50))
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    circle.uid, floor.uid = 7, 3
    circle.move_to(Matrix.vector([0, 8.5]))
    floor.move_to(Matrix.vector([0, 10]))
    collision = detect_collision(circle, floor)

    c0, c1, id0, id1 = find_contact_points(circle, floor, collision)

    assert c1 is None and id1 is None
    assert id0 == (7, 0)
    assert c0 is not None


def stack_engine() -> PhysicsEngine:
    """Build a three-box vertical stack resting on a static floor."""
    engine = PhysicsEngine(1200, 900,
                           DetectionKind.QUADTREE, show_contacts=False,
                           num_substeps=16)
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    for y in (7.9, 5.9, 3.9):
        box = Polygon.create_rectangle(4, 2, 2.0, (200, 100, 50))
        engine.add_body(box.move_to(Matrix.vector([0, y])))
    return engine


def contact_ids_by_pair(engine: PhysicsEngine) -> dict:
    """Map each colliding body-pair to its frozenset of contact feature IDs."""
    engine.update_swept_aabbs(FRAME)
    engine.collisions.clear()
    engine.broad_phase()
    result = {}
    for a, b in engine.collisions:
        collision = detect_collision(a, b)
        if collision is None:
            continue
        _c0, _c1, id0, id1 = find_contact_points(a, b, collision)
        ids = frozenset(fid for fid in (id0, id1) if fid is not None)
        result[tuple(sorted((a.uid, b.uid)))] = ids
    return result


def test_feature_ids_are_stable_across_a_settling_pile():
    """A settled stack keeps the same contact feature IDs frame to frame."""
    engine = stack_engine()
    for _ in range(150):
        engine.step(FRAME)

    previous = contact_ids_by_pair(engine)
    persistent = 0
    churned = 0
    for _ in range(60):
        engine.step(FRAME)
        current = contact_ids_by_pair(engine)
        for key, ids in current.items():
            if key in previous:
                persistent += 1
                if ids != previous[key]:
                    churned += 1
        previous = current

    assert persistent > 0
    churn_rate = churned / persistent
    # At 16 sub-steps the pile converges to a fixed rest pose, so its feature IDs must stay put.
    assert churn_rate < 0.05, f"feature-ID churn rate too high: {churn_rate:.4f}"


def ref_points_segment_distances(points: Matrix, a: Matrix, b: Matrix) -> list:
    """Original per-edge point-to-segment squared distances, the parity oracle."""
    ab = b - a
    t = (points - a).vecdot(ab, axis=1) / ab.vecdot(ab)
    closest = a + t.clip(0, 1) @ ab
    return list((points - closest).magnitude_squared(axis=1))


def ref_scan_polygon_edges(edges, points, tol, state, source_uid):
    """Replay the original per-edge sweep against the shared running manifold."""
    num_edges = edges.rows
    for i in range(num_edges):
        v0 = edges[i]
        v1 = edges[(i + 1) % num_edges]
        distances = ref_points_segment_distances(points, v0, v1)
        scan_edge_points(points, distances, tol, state, source_uid)


def ref_find_contact_points_polygon_polygon(a, b, tol=1e-5):
    """Find contact points with the original per-edge distance build."""
    a_verts = a.transformed_vertices
    b_verts = b.transformed_vertices
    state = [float("inf"), None, None, None, None]
    ref_scan_polygon_edges(a_verts, b_verts, tol, state, b.uid)
    ref_scan_polygon_edges(b_verts, a_verts, tol, state, a.uid)
    return state[1], state[2], state[3], state[4]


def points_equal(p, q) -> bool:
    """Bit-exact equality on two optional contact points."""
    if p is None or q is None:
        return p is None and q is None

    return p.x == q.x and p.y == q.y


def random_polygon(rng: random.Random):
    """Build a randomly shaped, randomly posed polygon for contact fuzzing."""
    if rng.random() < 0.5:
        body = Polygon.create_rectangle(rng.uniform(0.6, 2.0),
                                        rng.uniform(0.6, 2.0), 1.0,
                                        (180, 90, 90))
    else:
        body = Polygon.create_regular_polygon(rng.choice([3, 5, 6]),
                                              rng.uniform(0.5, 1.3), 1.0,
                                              (90, 90, 180))
    body.angle.x = rng.uniform(-3.14159, 3.14159)
    return body


def test_batched_contact_points_match_reference():
    """Batched contact generation equals the per-edge reference bit-for-bit."""
    rng = random.Random(20260620)
    hits = 0
    for k in range(2000):
        a = random_polygon(rng).move_to(Matrix.vector([0.0, 0.0]))
        b = random_polygon(rng).move_to(
            Matrix.vector([rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6)]))
        a.uid, b.uid = 2 * k, 2 * k + 1
        a.update_needed_ = b.update_needed_ = True
        if detect_collision(a, b) is None:
            continue

        hits += 1
        rc0, rc1, rid0, rid1 = ref_find_contact_points_polygon_polygon(a, b)
        c0, c1, id0, id1 = find_contact_points_polygon_polygon(a, b)
        assert points_equal(rc0, c0)
        assert points_equal(rc1, c1)
        assert (rid0, rid1) == (id0, id1)

    assert hits > 0


# --- contact kinematics --------------------------------------------------------------------------


def test_contact_velocity_static_is_zero():
    """A static body's material points are at rest."""
    floor = make_static_box(0, 0)
    velocity = contact_velocity(floor, Matrix.vector([1, 1]))
    assert velocity.x == 0.0 and velocity.y == 0.0


def test_contact_velocity_dynamic_matches_formula():
    """Material-point velocity is v + omega x r = v + omega * perpendicular(r)."""
    body = make_circle(0, 0, vx=3.0, vy=-1.0, omega=2.0)
    # perpendicular([0, 2]) = [-2, 0]; omega * that = [-4, 0]; plus v = [-1, -1].
    velocity = contact_velocity(body, Matrix.vector([0, 2]))
    assert velocity.x == pytest.approx(-1.0)
    assert velocity.y == pytest.approx(-1.0)


def test_relative_normal_velocity_is_negative_when_approaching():
    """Two bodies closing along the normal have negative relative normal velocity."""
    a = make_circle(0, 0, vy=0.0)
    b = make_circle(0, 1.5, vy=-5.0)
    normal = Matrix.vector([0, 1])
    rel = relative_normal_velocity(a, b, Matrix.vector([0, 0]),
                                   Matrix.vector([0, 0]), normal)
    assert rel == pytest.approx(-5.0)


# --- contact construction ------------------------------------------------------------------------


def test_build_contacts_skips_static_static_pairs():
    """A pair where neither body is dynamic produces no constraints."""
    a = make_static_box(0, 0, width=4, height=4)
    b = make_static_box(1, 0, width=4, height=4)
    assert build_contacts([(a, b)], None) == []


def test_build_contacts_skips_non_penetrating_pairs():
    """Far-apart bodies generate no constraints."""
    a = make_circle(0, 0)
    b = make_circle(50, 0)
    assert build_contacts([(a, b)], None) == []


def test_build_contacts_emits_penetrating_with_raw_bias():
    """A penetrating pair yields constraints whose bias is the raw pre-solve normal velocity."""
    a = make_circle(0, 0)
    b = make_circle(1.5, 0, vx=-4.0)
    constraints = build_contacts([(a, b)], None)
    assert constraints
    for constraint in constraints:
        expected = relative_normal_velocity(
            constraint.a, constraint.b, constraint.r_a, constraint.r_b, constraint.normal)
        assert constraint.bias_velocity == pytest.approx(expected)
        assert constraint.depth > 0


def test_build_contacts_records_overlay_points():
    """When an overlay set is given, the contact points are recorded into it."""
    a = make_circle(0, 0)
    b = make_circle(1.5, 0)
    overlay = set()
    build_contacts([(a, b)], overlay)
    assert len(overlay) >= 1


@pytest.mark.parametrize("seed", range(60))
def test_broad_box_never_rejects_a_real_overlap(seed):
    """The bounding-circle broad-phase cull keeps every pair that actually collides."""
    rng = random.Random(seed)

    def build():
        """Build a random circle or rectangle placed near the origin."""
        if rng.random() < 0.5:
            body = Circle.create(rng.uniform(0.6, 1.4), 2.0, (200, 100, 50))
        else:
            body = Polygon.create_rectangle(rng.uniform(1.0, 2.4),
                                            rng.uniform(1.0, 2.4), 2.0, (50, 120, 200))
        body.physics = True
        body.move_to(Matrix.vector([rng.uniform(-2, 2), rng.uniform(-2, 2)]))
        body.rotate_to(rng.uniform(0, 6.28))
        return body

    a, b = build(), build()
    a.uid, b.uid = 0, 1
    box_a = broad_box(a)
    box_b = broad_box(b)
    collision = detect_collision(a, b)
    if collision is not None and collision.depth > 0:
        assert not box_a.disjoint(box_b)
