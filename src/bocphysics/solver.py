"""Batched sub-step integration core shared by the XPBD solver.

Description:
    integrate_block is a module-level free function over plain bodies, so both
    the engine step and the Jacobi solver call the exact same integration with
    no engine instance required.
"""

from typing import List

from bocpy import Matrix

from .bodies import RigidBody


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
