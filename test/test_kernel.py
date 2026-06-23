"""Tests for the colour-batched velocity solver (kernel.resolve_batched)."""

import random

from bocpy import Matrix
import pytest

from bocphysics import kernel, solver
from bocphysics.bodies import Circle
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine


def make_engine() -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                         DetectionKind.QUADTREE, show_contacts=False)


def make_body(uid, x, y, vx, vy, spin=0.0):
    """Build a movable unit circle at (x, y) with the given velocity and spin."""
    body = Circle.create(2.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([x, y]))
    body.physics = True
    body.uid = uid
    body.linear_velocity = Matrix.vector([vx, vy])
    body.angular_velocity = spin
    return body


def velocities(bodies):
    """Snapshot each body's (vx, vy, spin) as plain floats."""
    return [(b.linear_velocity.x, b.linear_velocity.y, b.angular_velocity)
            for b in bodies]


def serial_reference(physics, manifolds, iters):
    """Run the serial resolve_pair_list the colour kernel mirrors.

    resolve_pair_list is the serial chokepoint: accumulated PGS for FRICTION, the
    stock per-iteration sweep for ROTATION. The colour-batched kernel reproduces
    this same per-mode math in colour order, so the parity tests build their
    reference straight from it.
    """
    solver.resolve_pair_list(physics, manifolds, iters, batched=False)


def build_disjoint_pairs():
    """Two far-apart overlapping pairs: four bodies, no shared endpoint."""
    a = make_body(1, -50.0, 0.0, 3.0, 0.0)
    b = make_body(2, -47.0, 0.0, -3.0, 0.0)
    c = make_body(3, 50.0, 0.0, -2.0, 1.0)
    d = make_body(4, 53.0, 0.0, 2.0, -1.0)
    bodies = [a, b, c, d]
    manifolds = [solver.build_manifold(a, b, None),
                 solver.build_manifold(c, d, None)]
    return bodies, manifolds


def build_chain(n):
    """A horizontal chain of n overlapping bodies sharing successive endpoints."""
    bodies = [make_body(i + 1, i * 3.0, 0.0, (-1) ** i * 2.0, 0.0)
              for i in range(n)]
    manifolds = []
    for i in range(n - 1):
        manifold = solver.build_manifold(bodies[i], bodies[i + 1], None)
        assert manifold is not None
        manifolds.append(manifold)

    return bodies, manifolds


def test_single_colour_matches_serial_bit_exact():
    """One colour of disjoint manifolds reproduces resolve_pair_list exactly."""
    engine = make_engine()
    ref_bodies, ref_manifolds = build_disjoint_pairs()
    assert len(kernel.colour_manifolds(ref_manifolds)) == 1
    serial_reference(engine.physics, ref_manifolds, 4)
    reference = velocities(ref_bodies)

    cand_bodies, cand_manifolds = build_disjoint_pairs()
    kernel.resolve_batched(engine.physics, cand_manifolds, 4)
    candidate = velocities(cand_bodies)

    assert candidate == reference


def test_single_colour_rotation_mode_bit_exact():
    """ROTATION mode (normal only) also matches serial within one colour."""
    engine = PhysicsEngine(1200, 900, PhysicsMode.ROTATION,
                           DetectionKind.QUADTREE, show_contacts=False)
    ref_bodies, ref_manifolds = build_disjoint_pairs()
    serial_reference(engine.physics, ref_manifolds, 4)
    reference = velocities(ref_bodies)

    cand_bodies, cand_manifolds = build_disjoint_pairs()
    kernel.resolve_batched(engine.physics, cand_manifolds, 4)

    assert velocities(cand_bodies) == reference


def test_multi_colour_converges_into_band():
    """A shared-endpoint chain agrees with serial within a tight band at convergence."""
    engine = make_engine()
    ref_bodies, ref_manifolds = build_chain(5)
    assert len(kernel.colour_manifolds(ref_manifolds)) > 1
    serial_reference(engine.physics, ref_manifolds, 64)
    reference = velocities(ref_bodies)

    cand_bodies, cand_manifolds = build_chain(5)
    kernel.resolve_batched(engine.physics, cand_manifolds, 64)
    candidate = velocities(cand_bodies)

    for (rvx, rvy, rw), (cvx, cvy, cw) in zip(reference, candidate):
        assert cvx == pytest.approx(rvx, abs=1e-6)
        assert cvy == pytest.approx(rvy, abs=1e-6)
        assert cw == pytest.approx(rw, abs=1e-6)


def test_empty_manifolds_is_noop():
    """resolve_batched on no manifolds leaves the system untouched."""
    engine = make_engine()
    kernel.resolve_batched(engine.physics, [], 4)


def test_flag_off_keeps_serial_path():
    """With the flag off, resolve_pair_list takes the serial (non-batched) path."""
    engine = make_engine()
    assert solver.use_batched_solver is False
    ref_bodies, ref_manifolds = build_disjoint_pairs()
    serial_reference(engine.physics, ref_manifolds, 4)
    reference = velocities(ref_bodies)

    cand_bodies, cand_manifolds = build_disjoint_pairs()
    solver.resolve_pair_list(engine.physics, cand_manifolds, 4)
    assert velocities(cand_bodies) == reference


def test_flag_on_routes_to_batched_kernel():
    """With the flag on, a single colour still matches the serial result exactly."""
    engine = make_engine()
    ref_bodies, ref_manifolds = build_disjoint_pairs()
    serial_reference(engine.physics, ref_manifolds, 4)
    reference = velocities(ref_bodies)

    cand_bodies, cand_manifolds = build_disjoint_pairs()
    solver.use_batched_solver = True
    try:
        solver.resolve_pair_list(engine.physics, cand_manifolds, 4)
    finally:
        solver.use_batched_solver = False

    assert velocities(cand_bodies) == reference


def test_flag_on_basic_mode_falls_back_to_serial():
    """BASIC mode is unsupported by the kernel, so the flag must not divert it."""
    engine = PhysicsEngine(1200, 900, PhysicsMode.BASIC,
                           DetectionKind.QUADTREE, show_contacts=False)
    ref_bodies, ref_manifolds = build_disjoint_pairs()
    solver.resolve_pair_list(engine.physics, ref_manifolds, 4)
    reference = velocities(ref_bodies)

    cand_bodies, cand_manifolds = build_disjoint_pairs()
    solver.use_batched_solver = True
    try:
        solver.resolve_pair_list(engine.physics, cand_manifolds, 4)
    finally:
        solver.use_batched_solver = False

    assert velocities(cand_bodies) == reference


def test_colouring_is_body_disjoint_within_colour():
    """Every colour group has pairwise-disjoint movable bodies."""
    _bodies, manifolds = build_chain(6)
    for group in kernel.colour_manifolds(manifolds):
        seen = set()
        for a, b, _collision, _c0, _c1 in group:
            for body in (a, b):
                if body.physics:
                    assert body.uid not in seen
                    seen.add(body.uid)


def make_fuzz_pair(seed_a, seed_b):
    """Build one overlapping pair from two sampled (vx, vy, spin) velocities."""
    a = make_body(1, -50.0, 0.0, seed_a[0], seed_a[1], seed_a[2])
    b = make_body(2, -47.5, 0.0, seed_b[0], seed_b[1], seed_b[2])
    manifold = solver.build_manifold(a, b, None)
    return [a, b], manifold


def test_fuzz_single_colour_bit_exact():
    """Random disjoint pairs stay bit-exact against the serial solver."""
    engine = make_engine()
    rng = random.Random(20260609)
    for _ in range(40):
        seed_a = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-2, 2))
        seed_b = (rng.uniform(-4, 4), rng.uniform(-4, 4), rng.uniform(-2, 2))

        ref_bodies, ref_manifold = make_fuzz_pair(seed_a, seed_b)
        serial_reference(engine.physics, [ref_manifold], 4)
        reference = velocities(ref_bodies)

        cand_bodies, cand_manifold = make_fuzz_pair(seed_a, seed_b)
        kernel.resolve_batched(engine.physics, [cand_manifold], 4)

        assert velocities(cand_bodies) == reference
