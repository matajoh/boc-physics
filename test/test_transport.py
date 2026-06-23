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
