"""Tests for immutable body geometry rehydration."""

import random

from bocpy import Matrix
import pytest

from bocphysics import geometry, solver, transport
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import PhysicsMode
from bocphysics.physics import Physics


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
    """Produce one fixed numeric state tuple for body construction."""
    return (rng.random(), rng.uniform(-3, 3), rng.uniform(-3, 3),
            rng.uniform(0, 6.28), rng.uniform(-5, 5), rng.uniform(-5, 5),
            rng.uniform(-3, 3), rng.uniform(0.8, 1.4), rng.randint(3, 6),
            rng.uniform(1.4, 2.4), rng.uniform(1.4, 2.4), rng.uniform(1.0, 1.6))


def assert_same_state(a, b):
    """Assert two bodies share identical mutable state to the last bit."""
    assert a.position.x == b.position.x
    assert a.position.y == b.position.y
    assert a.linear_velocity.x == b.linear_velocity.x
    assert a.linear_velocity.y == b.linear_velocity.y
    assert a.angle == b.angle
    assert a.angular_velocity == b.angular_velocity


def test_build_shell_circle_matches_original():
    """A rehydrated circle shell carries the original's geometry and flags."""
    body = build_dynamic(random_state(random.Random(1)))
    body.uid = 7
    geom = geometry.body_geometry(body)
    shell = geometry.build_shell(geom)
    assert isinstance(shell, Circle)
    assert shell.physics is True
    assert shell.radius == body.radius
    assert shell.inv_mass == body.inv_mass
    assert shell.inv_inertia == body.inv_inertia


def test_build_shell_static_polygon_has_no_physics():
    """A static polygon shell rebuilds with its pose and no physics flag."""
    body = Polygon.create_rectangle(4.0, 2.0, 1.0, (90, 90, 90), is_static=True)
    body.physics = False
    body.collision = True
    body.render = True
    body.move_to(Matrix.vector([1.0, -2.0])).rotate_to(0.3)
    body.uid = 3
    shell = geometry.build_shell(geometry.body_geometry(body))
    assert isinstance(shell, Polygon)
    assert shell.physics is False
    assert shell.position.x == 1.0
    assert shell.position.y == -2.0
    assert shell.angle == 0.3


def test_shell_cache_reuses_one_object_per_uid():
    """The cache returns the same shell instance for a repeated uid."""
    body = build_dynamic(random_state(random.Random(2)))
    body.uid = 11
    geom = {11: geometry.body_geometry(body)}
    cache = geometry.ShellCache()
    first = cache.shells(geom, [11])[0]
    second = cache.shells(geom, [11])[0]
    assert first is second


def test_evict_retired_drops_uids_absent_from_geometry():
    """A version bump prunes shells whose uid left the geometry snapshot."""
    rng = random.Random(5)
    keep = build_dynamic(random_state(rng))
    keep.uid = 11
    gone = build_dynamic(random_state(rng))
    gone.uid = 22
    full = {11: geometry.body_geometry(keep), 22: geometry.body_geometry(gone)}
    cache = geometry.ShellCache()
    cache.shells(full, [11, 22])
    cache.evict_retired(full, 1)
    assert set(cache.shells_) == {11, 22}

    cache.evict_retired({11: full[11]}, 2)
    assert set(cache.shells_) == {11}


def test_evict_retired_is_a_noop_within_one_version():
    """Without a version bump the gate short-circuits and nothing is pruned."""
    rng = random.Random(6)
    keep = build_dynamic(random_state(rng))
    keep.uid = 11
    gone = build_dynamic(random_state(rng))
    gone.uid = 22
    full = {11: geometry.body_geometry(keep), 22: geometry.body_geometry(gone)}
    cache = geometry.ShellCache()
    cache.shells(full, [11, 22])
    cache.evict_retired(full, 1)
    cache.evict_retired({11: full[11]}, 1)
    assert set(cache.shells_) == {11, 22}


@pytest.mark.parametrize("seed", range(30))
def test_rehydrated_solve_matches_serial(seed):
    """Solving on rehydrated shells reproduces the serial solve bit-exactly."""
    rng = random.Random(seed)
    physics = Physics(PhysicsMode.FRICTION)
    gravity = Matrix.vector([0, 9.81])
    sub_dt = (1 / 60) / 4

    dynamics = [build_dynamic(random_state(rng)) for _ in range(rng.randint(2, 6))]
    static = Polygon.create_rectangle(12.0, 1.0, 1.0, (90, 90, 90), is_static=True)
    static.physics = False
    static.collision = True
    static.render = True
    static.move_to(Matrix.vector([0, 3.5]))
    bodies = dynamics + [static]
    for uid, body in enumerate(bodies):
        body.uid = uid

    pairs = [(bodies[i], bodies[j])
             for i in range(len(bodies)) for j in range(i + 1, len(bodies))]

    geom = geometry.build_geometry(bodies)
    block = transport.pack_state(dynamics)

    cache = geometry.ShellCache()
    dyn_shells = cache.shells(geom, [b.uid for b in dynamics])
    stat_shells = cache.shells(geom, [static.uid])
    transport.apply_state(dyn_shells, block)

    by_uid = {s.uid: s for s in dyn_shells + stat_shells}
    shell_pairs = [(by_uid[a.uid], by_uid[b.uid]) for a, b in pairs]

    solver.solve_group_substep(physics, dynamics, pairs, gravity, sub_dt, 4, 5, None)
    solver.solve_group_substep(physics, dyn_shells, shell_pairs,
                               gravity, sub_dt, 4, 5, None)

    for original, shell in zip(dynamics, dyn_shells):
        assert_same_state(original, shell)
