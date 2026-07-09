"""Batched sub-step integration core shared by the XPBD solver.

Description:
    integrate_block is a module-level free function over plain bodies, so both
    the engine step and the Jacobi solver call the exact same integration with
    no engine instance required.
"""

from typing import Optional

from bocpy import Matrix

from .bodies import RigidBody, RigidBodyState
from .contacts import build_contacts, ContactConstraint
from .physics import Physics


ACC_WIDTH = 4
EPS = 1e-9


Pair = tuple[RigidBody, RigidBody]
ContactSet = set[tuple[float, float]]


class Solver:
    """Batched Jacobi XPBD sub-step solver over one group of bodies."""

    def __init__(self, bodies: list[RigidBodyState], gravity: Matrix, physics: Physics):
        """Create a solver over the group's dynamic bodies with shared gravity and physics config."""        
        self.bodies = [body for body in bodies if body.physics]
        if len(bodies) == 0:
            self.pos_acc = None
            self.vel_acc = None
        else:
            self.pos_acc = Matrix.zeros((len(bodies), ACC_WIDTH))
            self.vel_acc = Matrix.zeros((len(bodies), ACC_WIDTH))

        self.row_of = {body.uid: i for i, body in enumerate(self.bodies)}
        self.pairs: list[Pair] = None
        self.constraints: list[ContactConstraint] = None
        self.prev_pose: dict[int, Matrix] = None
        self.gravity = gravity
        self.physics = physics

    @staticmethod
    def generalized_inverse_mass(body: RigidBody, r: Matrix, direction: Matrix) -> float:
        """Generalised inverse mass along direction: 1/m + (r x dir)^2 / I, zero for a static body."""
        if not body.physics:
            return 0.0
        rn = r.cross(direction)
        return body.inv_mass + rn * rn * body.inv_inertia

    def solve(self, dt: float, num_substeps: int,
              pairs: list[tuple[RigidBody, RigidBody]],
              contacts: Optional[ContactSet]):
        """Advance one group of bodies over all sub-steps with the Jacobi solver."""
        if not self.bodies:
            return

        self.pairs = pairs
        self.contacts = contacts
        sub_dt = dt / num_substeps
        for _ in range(num_substeps):
            self.solve_substep(sub_dt)

    def solve_substep(self, sub_dt: float):
        """Advance the dynamic bodies one Jacobi XPBD sub-step (accumulate then apply, twice)."""
        previous = self.snapshot_poses()
        self.integrate_block(sub_dt)
        self.constraints = build_contacts(self.pairs, self.contacts)
        self.prev_pose = {body.uid: pose for body, pose in zip(self.bodies, previous)}
        lambdas = self.accumulate_positions()
        self.apply_positions()
        Physics.derive_velocities(self.bodies, previous, sub_dt)
        self.accumulate_velocities(lambdas, sub_dt)
        self.apply_velocities()

    def apply_positions(self):
        """Apply each owned body's averaged position correction from its accumulator row."""
        for body in self.bodies:
            row = self.row_of.get(body.uid)
            if row is None:
                continue

            count = self.pos_acc[row, 3]
            if count == 0.0:
                continue

            body.move(self.pos_acc[row, :2] / count)
            body.rotate(self.pos_acc[row, 2] / count)
            body.update_transform()

    def accumulate_positions(self) -> list[float]:
        """Accumulate every contact's normal + static-friction position correction, frozen-pose.

        Description:
            Reads only the (frozen) poses carried by the constraints and prev_pose;
            writes nothing to the bodies, only into the owned rows of acc. Returns
            the normal lambda per constraint in order so the velocity pass can zip
            them, matching Gauss-Seidel :func:`bocphysics.solve_positions`.
        """
        self.pos_acc[:] = 0
        lambdas = []
        for constraint in self.constraints:
            update, lambda_n = self.physics.position_update(constraint, self.prev_pose)
            lambdas.append(lambda_n)
            if update is None:
                continue

            self.accumulate(self.pos_acc, constraint.a, update.da, update.count)
            self.accumulate(self.pos_acc, constraint.b, update.db, update.count)

        return lambdas

    def accumulate(self, acc: Matrix, body: RigidBody,
                   delta: tuple[Matrix, float], count: float):
        """Fold one contribution into a dynamic, owned body's accumulator row.

        Description:
            Skips a static body and any body ``row_of`` does not map. This is the
            whole "write only your own rows" rule.
        """
        if not body.physics:
            return
        row = self.row_of.get(body.uid)
        if row is None:
            return

        acc[row, :2] += delta[0]
        acc[row, 2] += delta[1]
        acc[row, 3] += count

    def accumulate_velocities(self, lambdas: list[float], dt: float):
        """Accumulate the dynamic-friction and restitution velocity impulses, frozen-velocity.

        Description:
            The same velocity pass as :func:`bocphysics.solve_velocities`, but
            each impulse is folded into the owned rows of acc rather than applied
            immediately, so every contact reads the velocity frozen at the start of
            this pass. The averaged apply then updates each body once.
        """
        g = self.gravity.magnitude()
        self.vel_acc[:] = 0
        for constraint, lambda_n in zip(self.constraints, lambdas):
            update = self.physics.velocity_update(constraint, g, lambda_n, dt)
            if update is None:
                continue

            self.accumulate(self.vel_acc, constraint.a, update.da, update.count)
            self.accumulate(self.vel_acc, constraint.b, update.db, update.count)

    def snapshot_poses(self) -> list[Matrix]:
        """Record each body's pose as (position copy, angle) so derive_velocities reads no aliased Matrix."""
        return [Matrix.concat([body.position, body.angle], axis=1) for body in self.bodies]

    def integrate_block(self, dt: float):
        """Integrate every dynamic body in one batched semi-implicit Euler step.

        Description:
            Gathers the bodies' velocities, positions, angles, and spins into
            (N x 2) and (N x 1) Matrix blocks, advances them with three batched
            ops, then scatters the rows back. This is bit-identical to calling
            body.step one at a time, but pays the per-element float cost in C.
        """
        n = len(self.bodies)
        if n == 0:
            return

        for body in self.bodies:
            body.step(dt, self.gravity)

        """
        # TODO revisit this
        velocity = Matrix.concat([b.linear_velocity for b in self.bodies])
        position = Matrix.concat([b.position for b in self.bodies])
        angle = Matrix.concat([b.angle for b in self.bodies])
        spin = Matrix.concat([b.angular_velocity for b in self.bodies])

        velocity += self.gravity * dt
        position.scaled_add(dt, velocity, in_place=True)
        angle.scaled_add(dt, spin, in_place=True)

        for i, body in enumerate(self.bodies):
            body.linear_velocity[:] = velocity[i]
            body.position[:] = position[i]
            body.angle.x = angle[i]
            body.update_transform()
        """

    def apply_velocities(self):
        """Apply each owned body's averaged velocity impulse from its accumulator row."""
        for body in self.bodies:
            row = self.row_of.get(body.uid)
            if row is None:
                continue
            count = self.vel_acc[row, 3]
            if count == 0.0:
                continue
            dlin = self.vel_acc[row, :2]
            lv = body.linear_velocity
            lv += dlin / count
            av = body.angular_velocity
            av += self.vel_acc[row, 2] / count
