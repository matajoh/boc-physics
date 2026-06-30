"""Tests for patch mutable-state block transport."""

import random

from bocpy import Matrix
import pytest

from bocphysics import transport
from bocphysics.bodies import Circle, Polygon


def build_body(state):
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


def random_states(rng, count):
    """Produce count fixed numeric state tuples for body construction."""
    return [(rng.random(), rng.uniform(-12, 12), rng.uniform(-12, 6),
             rng.uniform(0, 6.28), rng.uniform(-5, 5), rng.uniform(-5, 5),
             rng.uniform(-3, 3), rng.uniform(0.6, 1.2), rng.randint(3, 6),
             rng.uniform(1.2, 2.2), rng.uniform(1.2, 2.2), rng.uniform(0.8, 1.3))
            for _ in range(count)]


@pytest.mark.parametrize("seed", range(30))
def test_pack_then_apply_restores_state_bit_exact(seed):
    """apply_state must undo any mutation, reproducing the packed state exactly."""
    rng = random.Random(seed)
    count = rng.randint(1, 12)
    states = random_states(rng, count)
    bodies = [build_body(s) for s in states]
    for i, body in enumerate(bodies):
        body.uid = i

    block = transport.pack_state(bodies)

    for body in bodies:
        body.move(Matrix.vector([7.0, -3.0]))
        body.linear_velocity = Matrix.vector([99.0, -99.0])
        body.angular_velocity = 42.0
        body.rotate_to(1.234)

    transport.apply_state(bodies, block)

    for body, state in zip(bodies, states):
        (_, x, y, ang, vx, vy, spin, *_rest) = state
        assert body.position.x == x
        assert body.position.y == y
        assert body.linear_velocity.x == vx
        assert body.linear_velocity.y == vy
        assert body.angle == ang
        assert body.angular_velocity == spin
        assert body.update_needed_ is True


@pytest.mark.parametrize("seed", range(30))
def test_packed_uids_match_body_order(seed):
    """The packed uid column mirrors the body order it was gathered from."""
    rng = random.Random(seed)
    count = rng.randint(1, 12)
    bodies = [build_body(s) for s in random_states(rng, count)]
    for i, body in enumerate(bodies):
        body.uid = i * 3 + 1

    block = transport.pack_state(bodies)
    assert transport.uids_of(block) == [body.uid for body in bodies]
    assert block.rows == count


def test_pack_state_block_shape():
    """A packed block carries one row per body and the seven named columns."""
    rng = random.Random(0)
    bodies = [build_body(s) for s in random_states(rng, 5)]
    for i, body in enumerate(bodies):
        body.uid = i
    block = transport.pack_state(bodies)
    assert (block.rows, block.columns) == (5, transport.WIDTH)


@pytest.mark.parametrize("seed", range(30))
def test_state_gather_scatter_round_trip(seed):
    """State.scatter must restore exactly what gather wrote, statics excluded."""
    rng = random.Random(seed)
    count = rng.randint(1, 12)
    states = random_states(rng, count)
    bodies = [build_body(s) for s in states]
    for i, body in enumerate(bodies):
        body.uid = i

    state = transport.State(bodies)
    assert state.row_of == {body.uid: i for i, body in enumerate(bodies)}

    for body in bodies:
        body.move(Matrix.vector([5.0, -2.0]))
        body.angular_velocity = 11.0

    state.scatter()

    for body, src in zip(bodies, states):
        (_, x, y, ang, vx, vy, spin, *_rest) = src
        assert body.position.x == x
        assert body.angular_velocity == spin


def test_state_excludes_statics():
    """Static bodies stay out of the pool; only dynamics get rows."""
    rng = random.Random(0)
    bodies = [build_body(s) for s in random_states(rng, 4)]
    for i, body in enumerate(bodies):
        body.uid = i
    bodies[1].physics = False
    state = transport.State(bodies)
    assert state.block.rows == 3
    assert 1 not in state.row_of


@pytest.mark.parametrize("seed", range(30))
def test_geometry_pool_rows_match_transformed_geometry(seed):
    """Each pool row reproduces its polygon's transformed verts/normals exactly."""
    rng = random.Random(seed)
    bodies = [build_body(s) for s in random_states(rng, rng.randint(2, 10))]
    for i, body in enumerate(bodies):
        body.uid = i
    polys = [b for b in bodies if isinstance(b, Polygon)]
    if not polys:
        pytest.skip("no polygons in this draw")

    pool = transport.GeometryPool(bodies)
    assert pool.geom_x.rows == len(polys)
    assert set(pool.row_of) == {p.uid for p in polys}

    for poly in polys:
        r = pool.row_of[poly.uid]
        pv = poly.transformed_vertices
        pn = poly.transformed_normals
        for v in range(pv.rows):
            assert pool.geom_x[r, v] == pv[v, 0]
            assert pool.geom_y[r, v] == pv[v, 1]
        for j in range(pn.rows):
            assert pool.norm_x[r, j] == pn[j, 0]
            assert pool.norm_y[r, j] == pn[j, 1]


@pytest.mark.parametrize("seed", range(30))
def test_geometry_pool_sync_from_matches_body_sync(seed):
    """sync_from(pose columns) reproduces the body-sourced sync bit-for-bit."""
    rng = random.Random(seed)
    bodies = [build_body(s) for s in random_states(rng, rng.randint(2, 10))]
    for i, body in enumerate(bodies):
        body.uid = i
    polys = [b for b in bodies if isinstance(b, Polygon)]
    if not polys:
        pytest.skip("no polygons in this draw")

    pool = transport.GeometryPool(bodies)
    px = [p.position.x for p in pool.polys]
    py = [p.position.y for p in pool.polys]
    angle = [p.angle for p in pool.polys]
    pool.sync_from(px, py, angle)

    reference = transport.GeometryPool(bodies)
    for r in range(pool.geom_x.rows):
        for v in range(pool.vmax):
            assert pool.geom_x[r, v] == reference.geom_x[r, v]
            assert pool.geom_y[r, v] == reference.geom_y[r, v]
        for j in range(pool.nmax):
            assert pool.norm_x[r, j] == reference.norm_x[r, j]
            assert pool.norm_y[r, j] == reference.norm_y[r, j]


@pytest.mark.parametrize("seed", range(30))
def test_geometry_pool_sync_from_block_matches_bodies(seed):
    """Refreshing the pool from a dynamics block matches body pose for dyn and static."""
    rng = random.Random(seed)
    bodies = [build_body(s) for s in random_states(rng, rng.randint(2, 10))]
    for i, body in enumerate(bodies):
        body.uid = i
    for body in bodies:
        if isinstance(body, Polygon) and rng.random() < 0.3:
            body.physics = False
    polys = [b for b in bodies if isinstance(b, Polygon)]
    if not polys:
        pytest.skip("no polygons in this draw")

    state = transport.State(bodies)
    pool = transport.GeometryPool(bodies)
    if state.block is not None:
        pool.sync_from_block(state.block, state.row_of)

    reference = transport.GeometryPool(bodies)
    for r in range(pool.geom_x.rows):
        for v in range(pool.vmax):
            assert pool.geom_x[r, v] == reference.geom_x[r, v]
            assert pool.geom_y[r, v] == reference.geom_y[r, v]
        for j in range(pool.nmax):
            assert pool.norm_x[r, j] == reference.norm_x[r, j]
            assert pool.norm_y[r, j] == reference.norm_y[r, j]


def test_geometry_pool_excludes_circles():
    """Circles have no geometry, so they never get a pool row."""
    rng = random.Random(0)
    bodies = [build_body(s) for s in random_states(rng, 8)]
    for i, body in enumerate(bodies):
        body.uid = i
    bodies[0] = Circle.create(1.0, 2.0, (10, 10, 10))
    bodies[0].uid = 0
    pool = transport.GeometryPool(bodies)
    assert bodies[0].uid not in pool.row_of


@pytest.mark.parametrize("seed", range(30))
def test_assert_block_mirrors_passes_for_packed_block(seed):
    """A block freshly packed from bodies mirrors them, so the guard never fires."""
    rng = random.Random(seed)
    bodies = [build_body(s) for s in random_states(rng, rng.randint(2, 10))]
    for i, body in enumerate(bodies):
        body.uid = i
    for body in bodies:
        if rng.random() < 0.3:
            body.physics = False

    state = transport.State(bodies)
    if state.block is None:
        pytest.skip("no dynamic bodies in this draw")

    transport.assert_block_mirrors(state.block, state.row_of, bodies)


def test_assert_block_mirrors_catches_divergence():
    """Perturbing one block row trips the mirror guard."""
    rng = random.Random(0)
    bodies = [build_body(s) for s in random_states(rng, 6)]
    for i, body in enumerate(bodies):
        body.uid = i
        body.physics = True

    state = transport.State(bodies)
    state.block[0, transport.POSITION.start] += 1.0
    with pytest.raises(AssertionError):
        transport.assert_block_mirrors(state.block, state.row_of, bodies)
