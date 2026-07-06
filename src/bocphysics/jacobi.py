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

import math
from typing import Dict, List, Tuple

from bocpy import Matrix

from . import xpbd
from .bodies import RigidBody
from .physics import Physics
from .solver import integrate_block

# Accumulator columns: summed linear delta (x, y), summed scalar delta, contribution count.
ACC_WIDTH = 4

# A uid/id -> block-row map identifying which rows a behavior owns and may write.
RowMap = Dict[int, int]


def new_accumulator(n: int) -> Matrix:
    """Zeroed ``(n x 4)`` accumulator block; rows are [delta_x, delta_y, delta_scalar, count]."""
    return Matrix.zeros((n, ACC_WIDTH))


def _accumulate(acc: Matrix, row_of: RowMap, body: RigidBody, delta: Matrix, dscalar: float):
    """Fold one contribution into a dynamic, owned body's accumulator row.

    Description:
        Skips a static body and any body ``row_of`` does not map. This is the
        whole "write only your own rows" rule.
    """
    if not body.physics:
        return
    row = row_of.get(id(body))
    if row is None:
        return
    acc[row, 0] += delta.x
    acc[row, 1] += delta.y
    acc[row, 2] += dscalar
    acc[row, 3] += 1.0


def _accumulate_static_friction(acc: Matrix, row_of: RowMap, a: RigidBody, b: RigidBody,
                                normal: Matrix, r_a: Matrix, r_b: Matrix,
                                lambda_n: float, mu_s: float, prev_pose: dict):
    """Accumulate the static-friction position correction while inside the cone.

    Description:
        Identical physics to :func:`bocphysics.xpbd._apply_static_friction` --
        cancel the contact's tangential slip since the sub-step start while the
        tangential lambda stays within ``mu_s * lambda_n`` -- but the correction
        is folded into the accumulator instead of mutating the bodies, so it
        composes with the normal push in one averaged apply.
    """
    slip = (xpbd._contact_point_slip(a, r_a, prev_pose)
            - xpbd._contact_point_slip(b, r_b, prev_pose))
    slip_t = slip - normal * slip.vecdot(normal)
    mag = math.sqrt(slip_t.vecdot(slip_t))
    if mag < xpbd.EPS:
        return
    t = slip_t / mag
    w_t = xpbd.generalized_inverse_mass(a, r_a, t) + xpbd.generalized_inverse_mass(b, r_b, t)
    if w_t < xpbd.EPS or mag / w_t > mu_s * lambda_n:
        return
    p = slip_t * (1.0 / w_t)
    _accumulate(acc, row_of, a, p * -a.inv_mass, -r_a.cross(p) * a.inv_inertia)
    _accumulate(acc, row_of, b, p * b.inv_mass, r_b.cross(p) * b.inv_inertia)


def accumulate_positions(constraints: List[xpbd.ContactConstraint], physics: Physics,
                         prev_pose: dict, row_of: RowMap, acc: Matrix) -> List[float]:
    """Accumulate every contact's normal + static-friction position correction, frozen-pose.

    Description:
        Reads only the (frozen) poses carried by the constraints and prev_pose;
        writes nothing to the bodies, only into the owned rows of acc. Returns
        the normal lambda per constraint in order so the velocity pass can zip
        them, matching Gauss-Seidel :func:`bocphysics.xpbd.solve_positions`.
    """
    lambdas = []
    for a, b, normal, r_a, r_b, depth, _bias in constraints:
        w = xpbd.generalized_inverse_mass(a, r_a, normal) + xpbd.generalized_inverse_mass(b, r_b, normal)
        if w < xpbd.EPS:
            lambdas.append(0.0)
            continue
        magnitude = depth / w
        lambdas.append(magnitude)
        p = normal * magnitude
        _accumulate(acc, row_of, a, p * -a.inv_mass, -r_a.cross(p) * a.inv_inertia)
        _accumulate(acc, row_of, b, p * b.inv_mass, r_b.cross(p) * b.inv_inertia)
        _accumulate_static_friction(acc, row_of, a, b, normal, r_a, r_b, magnitude,
                                    physics.static_friction, prev_pose)
    return lambdas


def apply_positions(bodies: List[RigidBody], row_of: RowMap, acc: Matrix):
    """Apply each owned body's averaged position correction from its accumulator row."""
    for body in bodies:
        row = row_of.get(id(body))
        if row is None:
            continue
        count = acc[row, 3]
        if count == 0.0:
            continue
        delta = Matrix.vector([acc[row, 0], acc[row, 1]])
        body.move(delta * (1.0 / count))
        body.rotate_to(body.angle + acc[row, 2] / count)


def accumulate_velocities(physics: Physics, constraints: List[xpbd.ContactConstraint],
                          lambdas: List[float], h: float, gravity: Matrix,
                          row_of: RowMap, acc: Matrix):
    """Accumulate the dynamic-friction and restitution velocity impulses, frozen-velocity.

    Description:
        The same velocity pass as :func:`bocphysics.xpbd.solve_velocities`, but
        each impulse is folded into the owned rows of acc rather than applied
        immediately, so every contact reads the velocity frozen at the start of
        this pass. The averaged apply then updates each body once.
    """
    g = gravity.magnitude()
    for constraint, lam_n in zip(constraints, lambdas):
        a, b, normal, r_a, r_b, _depth, bias_velocity = constraint
        v = xpbd.contact_velocity(b, r_b) - xpbd.contact_velocity(a, r_a)
        vn = v.vecdot(normal)
        vt = v - normal * vn
        vt_mag = math.sqrt(vt.vecdot(vt))
        if vt_mag > xpbd.EPS:
            f_n = lam_n / (h * h)
            t = vt / vt_mag
            w_t = xpbd.generalized_inverse_mass(a, r_a, t) + xpbd.generalized_inverse_mass(b, r_b, t)
            if w_t > xpbd.EPS:
                dvt = -min(h * physics.dynamic_friction * f_n, vt_mag)
                _add_impulse(acc, row_of, a, b, r_a, r_b, t * (dvt / w_t))
        e = 0.0 if abs(vn) <= 2 * g * h else physics.restitution
        w_n = xpbd.generalized_inverse_mass(a, r_a, normal) + xpbd.generalized_inverse_mass(b, r_b, normal)
        if w_n > xpbd.EPS:
            dvn = -vn + max(-e * bias_velocity, 0.0)
            _add_impulse(acc, row_of, a, b, r_a, r_b, normal * (dvn / w_n))


def _add_impulse(acc: Matrix, row_of: RowMap, a: RigidBody, b: RigidBody,
                 r_a: Matrix, r_b: Matrix, impulse: Matrix):
    """Fold one velocity impulse into both bodies' owned rows (a gets -, b gets +)."""
    _accumulate(acc, row_of, a, impulse * -a.inv_mass, -r_a.cross(impulse) * a.inv_inertia)
    _accumulate(acc, row_of, b, impulse * b.inv_mass, r_b.cross(impulse) * b.inv_inertia)


def apply_velocities(bodies: List[RigidBody], row_of: RowMap, acc: Matrix):
    """Apply each owned body's averaged velocity impulse from its accumulator row."""
    for body in bodies:
        row = row_of.get(id(body))
        if row is None:
            continue
        count = acc[row, 3]
        if count == 0.0:
            continue
        dlin = Matrix.vector([acc[row, 0], acc[row, 1]])
        body.linear_velocity = body.linear_velocity + dlin * (1.0 / count)
        body.angular_velocity = body.angular_velocity + acc[row, 2] / count


def solve_substep(physics: Physics, bodies: List[RigidBody],
                  pairs: List[Tuple[RigidBody, RigidBody]], gravity: Matrix,
                  sub_dt: float, contacts: xpbd.ContactSet = None):
    """Advance the dynamic bodies one Jacobi XPBD sub-step (accumulate then apply, twice)."""
    if not bodies:
        return
    previous = xpbd.snapshot_poses(bodies)
    integrate_block(bodies, gravity, sub_dt)
    constraints = xpbd.build_contacts(pairs, contacts)
    prev_pose = {id(body): pose for body, pose in zip(bodies, previous)}
    row_of = {id(body): i for i, body in enumerate(bodies)}
    pos_acc = new_accumulator(len(bodies))
    lambdas = accumulate_positions(constraints, physics, prev_pose, row_of, pos_acc)
    apply_positions(bodies, row_of, pos_acc)
    xpbd.derive_velocities(bodies, previous, sub_dt)
    vel_acc = new_accumulator(len(bodies))
    accumulate_velocities(physics, constraints, lambdas, sub_dt, gravity, row_of, vel_acc)
    apply_velocities(bodies, row_of, vel_acc)


def solve_group_substep(physics: Physics, bodies: List[RigidBody],
                        pairs: List[Tuple[RigidBody, RigidBody]], gravity: Matrix,
                        sub_dt: float, num_substeps: int, contacts: xpbd.ContactSet = None):
    """Advance one group of bodies over all sub-steps with the Jacobi solver."""
    for _ in range(num_substeps):
        solve_substep(physics, bodies, pairs, gravity, sub_dt, contacts)
