"""Serial 2D XPBD contact solver: position-based rigid-body dynamics.

Description:
    A faithful 2D specialisation of Mueller et al. 2020, "Detailed Rigid Body
    Simulation with Extended Position Based Dynamics" (Algorithm 2). Orientation
    is a scalar angle, inertia a scalar, and the cross products r x n / r x p
    collapse to scalars. Each sub-step integrates the bodies once, re-evaluates
    the narrow phase at the new pose, runs a single Gauss-Seidel position pass
    (numPosIters = 1, compliance alpha = 0 for rigid contacts), derives the
    velocity from the position delta, then applies one velocity pass for dynamic
    Coulomb friction (Eqn 30) and restitution (Eqn 34). These are module-level
    free functions over plain bodies, pairs, and a physics system, so a worker
    sub-interpreter can run the identical solve -- "same core, different
    scheduler". The integrator is shared with the impulse path via
    solver.integrate_block; the import is strictly one-way (solver never imports
    this module), so no cycle is formed.
"""

import math
from typing import List, NamedTuple, Optional, Set, Tuple

from bocpy import Matrix

from .bodies import Circle, RigidBody
from .collisions import (batched_circle_circle, batched_circle_polygon,
                         detect_collision)
from .contacts import find_contact_points
from .physics import Physics
from .solver import integrate_block

ContactSet = Optional[Set[Tuple[float, float]]]

# Effective-mass and tangent-speed floor below which a contact contributes no impulse.
EPS = 1e-9


class ContactConstraint(NamedTuple):
    """One contact-point constraint shared by the position and velocity passes."""

    a: RigidBody
    b: RigidBody
    normal: Matrix
    r_a: Matrix
    r_b: Matrix
    depth: float
    bias_velocity: float


def generalized_inverse_mass(body: RigidBody, r: Matrix, direction: Matrix) -> float:
    """Generalised inverse mass along direction: 1/m + (r x dir)^2 / I, zero for a static body."""
    if not body.physics:
        return 0.0
    rn = r.cross(direction)
    return body.inv_mass + rn * rn * body.inv_inertia


def contact_velocity(body: RigidBody, r: Matrix) -> Matrix:
    """World velocity of the material point at anchor r: v + omega x r, zero for a static body."""
    if not body.physics:
        return Matrix.vector([0, 0])
    return body.linear_velocity + body.angular_velocity * r.perpendicular()


def relative_normal_velocity(a: RigidBody, b: RigidBody, r_a: Matrix,
                             r_b: Matrix, normal: Matrix) -> float:
    """Pre-solve relative velocity along the contact normal (b relative to a)."""
    return (contact_velocity(b, r_b) - contact_velocity(a, r_a)).vecdot(normal)


def _batch_circle_collisions(pairs):
    """Resolve circle-circle and circle-poly pairs in two batched SAT calls.

    Returns a dict mapping each eligible pair's index to its Collision-or-None,
    in the same orientation detect_collision would yield. Poly-poly pairs are
    absent so the caller falls back to the per-pair test for them.
    """
    cc_idx, cc = [], []
    cp_idx, cp, cp_flip = [], [], []
    for i, (a, b) in enumerate(pairs):
        if isinstance(a, Circle) and isinstance(b, Circle):
            cc_idx.append(i)
            cc.append((a, b))
        elif isinstance(a, Circle):
            cp_idx.append(i)
            cp.append((a, b))
            cp_flip.append(False)
        elif isinstance(b, Circle):
            cp_idx.append(i)
            cp.append((b, a))
            cp_flip.append(True)
    out = {}
    for i, col in zip(cc_idx, batched_circle_circle(cc)) if cc else ():
        out[i] = col
    for i, flip, col in zip(cp_idx, cp_flip, batched_circle_polygon(cp)) if cp else ():
        out[i] = col.reverse() if (flip and col is not None) else col
    return out


def build_contacts(pairs: List[Tuple[RigidBody, RigidBody]],
                   contacts: ContactSet = None) -> List[ContactConstraint]:
    """Re-evaluate the narrow phase at the current pose; one constraint per penetrating contact point.

    Description:
        Pairs where neither body is dynamic are skipped (a static-static contact
        moves nothing and feeds a zero effective mass into the solve). Only
        penetrating collisions (depth > 0) emit constraints, so every returned
        constraint yields a position lambda the velocity pass can reuse. The
        bias velocity is the raw pre-solve normal velocity; restitution is
        applied later in solve_velocities, not folded in here. When contacts is
        not None, the contact points are recorded for the show-contacts overlay.
        Pairs whose AABBs are disjoint are rejected before the full SAT; the box
        test is conservative (it never rejects a real overlap), so the emitted
        constraint set is identical to running detect_collision on every pair.
    """
    constraints = []
    eligible = [(a, b) for a, b in pairs
                if (a.physics or b.physics) and not a.aabb.disjoint(b.aabb)]
    resolved = _batch_circle_collisions(eligible)
    for i, (a, b) in enumerate(eligible):
        collision = resolved[i] if i in resolved else detect_collision(a, b)
        if collision is None or collision.depth <= 0:
            continue
        normal = collision.normal
        c0, c1, _id0, _id1 = find_contact_points(a, b, collision)
        if contacts is not None:
            contacts.add((c0.x, c0.y))
            if c1 is not None:
                contacts.add((c1.x, c1.y))
        for point in (c0, c1):
            if point is None:
                continue
            r_a = point - a.position
            r_b = point - b.position
            bias_velocity = relative_normal_velocity(a, b, r_a, r_b, normal)
            constraints.append(ContactConstraint(a, b, normal, r_a, r_b, collision.depth, bias_velocity))
    return constraints


def apply_positional_impulse(a: RigidBody, b: RigidBody, r_a: Matrix,
                             r_b: Matrix, impulse: Matrix):
    """Apply a positional impulse at the contact: a moves -impulse, b moves +impulse (mass-weighted)."""
    if a.physics:
        a.move(impulse * -a.inv_mass)
        a.rotate_to(a.angle - r_a.cross(impulse) * a.inv_inertia)
    if b.physics:
        b.move(impulse * b.inv_mass)
        b.rotate_to(b.angle + r_b.cross(impulse) * b.inv_inertia)


def solve_positions(constraints: List[ContactConstraint]) -> List[float]:
    """One Gauss-Seidel position pass; return the normal lambda per constraint, in order.

    Description:
        Each penetrating contact is pushed apart along its normal by depth / w,
        where w is the summed generalised inverse mass (compliance alpha = 0).
        The returned lambda feeds the friction bound in solve_velocities; the
        list is one-to-one with constraints so the velocity pass can zip them.
    """
    lambdas = []
    for a, b, normal, r_a, r_b, depth, _bias in constraints:
        w = generalized_inverse_mass(a, r_a, normal) + generalized_inverse_mass(b, r_b, normal)
        if w < EPS:
            lambdas.append(0.0)
            continue
        magnitude = depth / w
        lambdas.append(magnitude)
        apply_positional_impulse(a, b, r_a, r_b, normal * magnitude)
    return lambdas


def snapshot_poses(bodies: List[RigidBody]) -> List[Tuple[float, float, float]]:
    """Record each body's pose as scalar (x, y, angle) so derive_velocities reads no aliased Matrix."""
    return [(body.position.x, body.position.y, body.angle) for body in bodies]


def derive_velocities(bodies: List[RigidBody],
                      previous: List[Tuple[float, float, float]], h: float):
    """Set each body's velocity from its position delta over the sub-step (the XPBD velocity update)."""
    for body, (px, py, pa) in zip(bodies, previous):
        body.linear_velocity = Matrix.vector([(body.position.x - px) / h, (body.position.y - py) / h])
        body.angular_velocity = (body.angle - pa) / h


def apply_velocity_impulse(a: RigidBody, b: RigidBody, r_a: Matrix,
                           r_b: Matrix, impulse: Matrix):
    """Apply a velocity impulse at the contact: a gets -impulse, b gets +impulse (mass-weighted)."""
    if a.physics:
        a.linear_velocity = a.linear_velocity.scaled_add(-a.inv_mass, impulse)
        a.angular_velocity -= r_a.cross(impulse) * a.inv_inertia
    if b.physics:
        b.linear_velocity = b.linear_velocity.scaled_add(b.inv_mass, impulse)
        b.angular_velocity += r_b.cross(impulse) * b.inv_inertia


def solve_velocities(physics: Physics, constraints: List[ContactConstraint],
                     lambdas: List[float], h: float, gravity: Matrix):
    """One velocity pass: dynamic Coulomb friction (Eqn 30) then restitution (Eqn 34).

    Description:
        Friction caps the tangential change at the Coulomb bound mu_d * f_n,
        where the normal force f_n = lambda_n / h^2 comes from the position
        pass. Restitution adds back -e * bias_velocity along the normal, with
        the bounce gated off (e = 0) when the approach speed is at the gravity
        scale 2 * g * h to keep resting stacks from jittering. The contact
        normal points a -> b, so an approaching contact has bias_velocity < 0
        and the rebound is the positive branch max(-e * bias_velocity, 0).
    """
    g = gravity.magnitude()
    for constraint, lam_n in zip(constraints, lambdas):
        a, b, normal, r_a, r_b, _depth, bias_velocity = constraint
        v = contact_velocity(b, r_b) - contact_velocity(a, r_a)
        vn = v.vecdot(normal)
        vt = v - normal * vn
        vt_mag = math.sqrt(vt.vecdot(vt))
        if vt_mag > EPS:
            f_n = lam_n / (h * h)
            t = vt / vt_mag
            w_t = generalized_inverse_mass(a, r_a, t) + generalized_inverse_mass(b, r_b, t)
            if w_t > EPS:
                dvt = -min(h * physics.dynamic_friction * f_n, vt_mag)
                apply_velocity_impulse(a, b, r_a, r_b, t * (dvt / w_t))
        e = 0.0 if abs(vn) <= 2 * g * h else physics.restitution
        w_n = generalized_inverse_mass(a, r_a, normal) + generalized_inverse_mass(b, r_b, normal)
        if w_n > EPS:
            dvn = -vn + max(-e * bias_velocity, 0.0)
            apply_velocity_impulse(a, b, r_a, r_b, normal * (dvn / w_n))


def solve_substep(physics: Physics, bodies: List[RigidBody],
                  pairs: List[Tuple[RigidBody, RigidBody]], gravity: Matrix,
                  sub_dt: float, contacts: ContactSet = None):
    """Advance the dynamic bodies one XPBD sub-step: integrate, solve positions, derive, solve velocities."""
    previous = snapshot_poses(bodies)
    integrate_block(bodies, gravity, sub_dt)
    constraints = build_contacts(pairs, contacts)
    lambdas = solve_positions(constraints)
    derive_velocities(bodies, previous, sub_dt)
    solve_velocities(physics, constraints, lambdas, sub_dt, gravity)


def solve_group_substep(physics: Physics, bodies: List[RigidBody],
                        pairs: List[Tuple[RigidBody, RigidBody]], gravity: Matrix,
                        sub_dt: float, num_substeps: int, contacts: ContactSet = None):
    """Advance one group of bodies over all sub-steps with the XPBD solver."""
    for _ in range(num_substeps):
        solve_substep(physics, bodies, pairs, gravity, sub_dt, contacts)
