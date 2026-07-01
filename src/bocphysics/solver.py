"""Shared sub-step integration core, callable by the serial engine and parallel workers.

Description:
    integrate_block is a module-level free function over plain bodies, so a
    worker sub-interpreter can run the exact same integration as the serial
    engine -- "same core, different scheduler". No engine instance is required,
    which is what lets the parallel path reuse it verbatim.
"""

from typing import List

from bocpy import Matrix

from . import transport
from .bodies import RigidBody


# Module-global A/B toggle; the parallel path snapshots it once in begin(), so set it before then.
use_batched_solver = False


def integrate_block(bodies: List[RigidBody], gravity, dt: float):
    """Integrate every dynamic body in one batched semi-implicit Euler step.

    Description:
        Gathers the bodies' velocities, positions, angles, and spins into
        (N x 2) and (N x 1) Matrix blocks, advances them with three batched
        ops, then scatters the rows back. This is bit-identical to calling
        body.step one at a time, but pays the per-element float cost in C.
    """
    n = len(bodies)
    if n == 0:
        return

    velocity = Matrix(n, 2, [c for b in bodies
                             for c in (b.linear_velocity.x, b.linear_velocity.y)])
    position = Matrix(n, 2, [c for b in bodies
                             for c in (b.position.x, b.position.y)])
    angle = Matrix(n, 1, [b.angle for b in bodies])
    spin = Matrix(n, 1, [b.angular_velocity for b in bodies])

    velocity.add(gravity * dt, out=velocity)
    position.scaled_add(dt, velocity, in_place=True)
    angle.scaled_add(dt, spin, in_place=True)

    for i, body in enumerate(bodies):
        body.linear_velocity = velocity[i]
        body.position = position[i]
        body.angle = angle[i, 0]
        body.update_needed_ = True


def integrate_block_state(block, bodies: List[RigidBody], gravity, dt: float):
    """Integrate the dynamics block in place, mirroring the row-aligned bodies.

    Description:
        Same batched semi-implicit Euler as integrate_block, but the velocity,
        position, angle, and spin sub-blocks are read straight from the State
        block columns and written back into them, rather than gathered from the
        bodies. The bodies are still mirrored row-for-row so the scalar read
        surface stays in lockstep during the B-bridge; the result is
        bit-identical to integrate_block because the block mirrors the bodies on
        entry. bodies must be in block-row order (state.bodies).
    """
    n = block.rows
    if n == 0:
        return

    angle_col = slice(transport.ANGLE, transport.ANGLE + 1)
    spin_col = slice(transport.SPIN, transport.SPIN + 1)
    velocity = block[:, transport.VELOCITY]
    position = block[:, transport.POSITION]
    angle = block[:, angle_col]
    spin = block[:, spin_col]

    velocity += dt * gravity
    position.scaled_add(dt, velocity, in_place=True)
    angle.scaled_add(dt, spin, in_place=True)

    block[:, transport.VELOCITY] = velocity
    block[:, transport.POSITION] = position
    block[:, angle_col] = angle

    for i, body in enumerate(bodies):
        body.linear_velocity = velocity[i]
        body.position = position[i]
        body.angle = angle[i, 0]
        body.update_needed_ = True
