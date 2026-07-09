"""Jacobi contact solve: order-independent XPBD projection.

Description:
    This is the engine's serial XPBD contact solver. A sequential Gauss-Seidel
    projection mutates each body the moment a contact is resolved, so the next
    contact sees the updated pose; that order dependence drifts resting stacks
    (Seabra et al. 2023, arXiv 2311.09327). This module instead solves every
    contact the Jacobi way: each correction is computed from the frozen
    start-of-sub-step pose and merely accumulated per body, then the per-body
    average is applied in a second pass. The result is order-independent, so it
    holds stacks far better. It reuses the shared XPBD primitives in
    :mod:`bocphysics.xpbd` (narrow phase, position and velocity passes, friction).

    The accumulator is an ``(N x 4)`` Matrix -- one row per body, columns
    ``[delta_x, delta_y, delta_scalar, count]``. A contribution is folded into a
    body's row only when ``row_of`` maps it, so statics are skipped. Dividing
    each row by its count on apply is the position form of Tonge et al. 2012
    mass splitting, which keeps the Jacobi solve jitter-free.
"""

from typing import NamedTuple, Optional

from bocpy import Matrix

from bocphysics.geometry import contact_point_slip
from .bodies import RigidBody
from .contacts import contact_velocity, ContactConstraint

# Accumulator columns: summed linear delta (x, y), summed scalar delta, contribution count.
ACC_WIDTH = 4

# A uid/id -> block-row map identifying which rows a behavior owns and may write.
RowMap = dict[int, int]


ContactSet = Optional[set[tuple[float, float]]]

# Effective-mass and tangent-speed floor below which a contact contributes no impulse.
EPS = 1e-9


class ConstraintUpdate(NamedTuple("ConstraintUpdate", [("da", tuple[Matrix, float]),
                                                       ("db", tuple[Matrix, float]),
                                                       ("count", int)])):
    """Per-body accumulator rows (linear delta, scalar delta, count) for one contact."""

    @staticmethod
    def create(constraint: ContactConstraint, impulse: Matrix, count: float):
        """Build the per-body accumulator rows from one contact impulse and its contribution count."""
        da_vel = impulse * -constraint.a.inv_mass
        da_spin = -constraint.r_a.cross(impulse) * constraint.a.inv_inertia
        db_vel = impulse * constraint.b.inv_mass
        db_spin = constraint.r_b.cross(impulse) * constraint.b.inv_inertia
        da = da_vel, da_spin
        db = db_vel, db_spin
        return ConstraintUpdate(da, db, count)


class Physics(NamedTuple):
    """Immutable physics-config snapshot for the contact solve.

    Description:
        A body's ``.physics`` flag -- not this type -- decides which bodies the
        solver integrates and pushes; this snapshot only carries the friction and
        restitution coefficients the XPBD solve reads. It is an immutable
        NamedTuple of plain values shared by every sub-step of a frame.
    """

    restitution: float = 0.5
    static_friction: float = 0.5
    dynamic_friction: float = 0.5

    @staticmethod
    def inv_mass(body: RigidBody, r: Matrix, direction: Matrix) -> float:
        """Generalised inverse mass along direction: 1/m + (r x dir)^2 / I, zero for a static body."""
        if not body.physics:
            return 0.0
        rn = r.cross(direction)
        return body.inv_mass + rn * rn * body.inv_inertia

    @staticmethod
    def derive_velocities(bodies: list[RigidBody],
                          previous: list[tuple[Matrix, float]], h: float):
        """Set each body's velocity from its position delta over the sub-step (the XPBD velocity update)."""
        for body, prev in zip(bodies, previous):
            lv = body.linear_velocity
            Matrix.subtract(body.position, prev[0, :2], out=lv)
            lv /= h

            av = body.angular_velocity
            Matrix.subtract(body.angle, prev[0, 2], out=av)
            av /= h

    def position_update(self, constraint: ContactConstraint,
                        previous: list[tuple[Matrix, float]]) -> tuple[ConstraintUpdate, float]:
        """Compute one contact's normal + static-friction position correction and its normal lambda."""
        a, b, normal, r_a, r_b, depth, _ = constraint
        w = self.inv_mass(a, r_a, normal) + self.inv_mass(b, r_b, normal)
        if w < EPS:
            return None, 0.0

        magnitude = depth / w
        impulse = normal * magnitude
        count = 1

        if self.static_friction:
            prev_a = previous.get(a.uid)
            prev_b = previous.get(b.uid)
            count += self.add_static_friction(impulse, constraint, prev_a, prev_b, magnitude)

        return ConstraintUpdate.create(constraint, impulse, count), magnitude

    def add_static_friction(self, impulse: Matrix, constraint: ContactConstraint,
                            prev_a: Optional[Matrix], prev_b: Optional[Matrix],
                            lambda_n: float):
        """Fold the static-friction tangent correction into impulse; return 1 if applied, else 0."""
        a, b, normal, r_a, r_b, _, _ = constraint
        slip = contact_point_slip(a, r_a, prev_a) - contact_point_slip(b, r_b, prev_b)
        slip_t = slip - normal * slip.vecdot(normal)
        mag = slip_t.magnitude()
        if mag < EPS:
            return 0

        t = slip_t / mag
        w_t = self.inv_mass(a, r_a, t) + self.inv_mass(b, r_b, t)
        if w_t < EPS or mag / w_t > self.static_friction * lambda_n:
            return 0

        impulse += slip_t * (1.0 / w_t)
        return 1

    def velocity_update(self, constraint: ContactConstraint, g: float, lambda_n: float, h: float):
        """Compute one contact's dynamic-friction + restitution velocity impulse, frozen-velocity."""
        a, b, normal, r_a, r_b, _, bias_velocity = constraint

        impulse = None
        count = 0
        v = contact_velocity(b, r_b) - contact_velocity(a, r_a)
        vn = v.vecdot(normal)
        vt = v - normal * vn
        vt_mag = vt.magnitude()
        if vt_mag > EPS:
            f_n = lambda_n / (h * h)
            t = vt / vt_mag
            w_t = self.inv_mass(a, r_a, t) + self.inv_mass(b, r_b, t)
            if w_t > EPS:
                dvt = -min(h * self.dynamic_friction * f_n, vt_mag)
                impulse = t * (dvt / w_t)
                count += 1
        e = 0.0 if abs(bias_velocity) <= 2 * g * h else self.restitution
        w_n = self.inv_mass(a, r_a, normal) + self.inv_mass(b, r_b, normal)
        if w_n > EPS:
            dvn = -vn + max(-e * bias_velocity, 0.0)
            imp_n = normal * (dvn / w_n)
            count += 1
            if impulse is None:
                impulse = imp_n
            else:
                impulse += imp_n

        if impulse is None:
            return None

        return ConstraintUpdate.create(constraint, impulse, count)
