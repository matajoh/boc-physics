"""Tests for the transformed-polygon GeometryPool used by the narrow phase."""

import random

from bocpy import Matrix
import pytest

from bocphysics import xpbd
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
def test_geometry_pool_rows_match_transformed_geometry(seed):
    """Each pool row reproduces its polygon's transformed verts/normals exactly."""
    rng = random.Random(seed)
    bodies = [build_body(s) for s in random_states(rng, rng.randint(2, 10))]
    for i, body in enumerate(bodies):
        body.uid = i
    polys = [b for b in bodies if isinstance(b, Polygon)]
    if not polys:
        pytest.skip("no polygons in this draw")

    pool = xpbd.GeometryPool(bodies)
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


def test_geometry_pool_excludes_circles():
    """Circles have no geometry, so they never get a pool row."""
    rng = random.Random(0)
    bodies = [build_body(s) for s in random_states(rng, 8)]
    for i, body in enumerate(bodies):
        body.uid = i
    bodies[0] = Circle.create(1.0, 2.0, (10, 10, 10))
    bodies[0].uid = 0
    pool = xpbd.GeometryPool(bodies)
    assert bodies[0].uid not in pool.row_of
