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

from . import transport
from .bodies import AABB, Circle, Polygon, RigidBody
from .collisions import (batched_circle_circle, batched_circle_polygon,
                         batched_polygon_polygon, detect_collision)
from .contacts import batched_contact_points, find_contact_points
from .physics import Physics
from .solver import integrate_block

ContactSet = Optional[Set[Tuple[float, float]]]

# Effective-mass and tangent-speed floor below which a contact contributes no impulse.
EPS = 1e-9

# A static body's material points are at rest; share one zero to avoid a per-call alloc.
_ZERO_VELOCITY = Matrix.vector([0.0, 0.0])


class ContactConstraint(NamedTuple):
    """One contact-point constraint shared by the position and velocity passes."""

    a: RigidBody
    b: RigidBody
    normal: Matrix
    r_a: Matrix
    r_b: Matrix
    depth: float
    bias_velocity: float
    idx_a: Optional[int]
    idx_b: Optional[int]


def generalized_inverse_mass(body: RigidBody, r: Matrix, direction: Matrix) -> float:
    """Generalised inverse mass along direction: 1/m + (r x dir)^2 / I, zero for a static body."""
    if not body.physics:
        return 0.0
    rn = r.cross(direction)
    return body.inv_mass + rn * rn * body.inv_inertia


def contact_velocity(body: RigidBody, r: Matrix) -> Matrix:
    """World velocity of the material point at anchor r: v + omega x r, zero for a static body."""
    if not body.physics:
        return _ZERO_VELOCITY
    return body.linear_velocity + body.angular_velocity * r.perpendicular()


def _broad_box(body: RigidBody, state: Optional["transport.State"]) -> AABB:
    """Conservative bounding-circle AABB; dynamic pose from the block, static pose from the body.

    Description:
        The broad-phase cull only needs a box that never shrinks below the true
        bounds, so the rotation-invariant bounding circle (centre +/- radius)
        suffices and needs no angle. A dynamic body reads its centre from the
        State block so build_contacts touches no dynamic body pose; a static body
        never integrates, so its scalar centre is always valid. The looser box
        only widens the cull -- it never rejects a real overlap -- so the emitted
        constraint set and its order are unchanged.
    """
    row = state.row_of.get(body.uid) if state is not None and body.physics else None
    if row is not None:
        x = state.block[row, transport.POSITION.start]
        y = state.block[row, transport.POSITION.start + 1]
    else:
        x, y = body.position.x, body.position.y
    rad = body.radius
    return AABB(x - rad, y - rad, x + rad, y + rad)


def relative_normal_velocity(a: RigidBody, b: RigidBody, r_a: Matrix,
                             r_b: Matrix, normal: Matrix) -> float:
    """Pre-solve relative velocity along the contact normal (b relative to a)."""
    return (contact_velocity(b, r_b) - contact_velocity(a, r_a)).vecdot(normal)


def _batch_circle_collisions(pairs, geom, state=None):
    """Resolve circle-circle and circle-poly pairs in two batched SAT calls.

    Returns a dict mapping each eligible pair's index to its Collision-or-None,
    in the same orientation detect_collision would yield. geom is the shared
    GeometryPool over every eligible polygon, reused by both batched SAT calls.
    When state is given, circle centres are sourced from the State block.
    """
    cc_idx, cc = [], []
    cp_idx, cp, cp_flip = [], [], []
    pp_idx, pp = [], []
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
        else:
            pp_idx.append(i)
            pp.append((a, b))
    out = {}
    for i, col in zip(cc_idx, batched_circle_circle(cc, state)) if cc else ():
        out[i] = col
    if cp:
        for i, flip, col in zip(cp_idx, cp_flip, batched_circle_polygon(cp, geom, state)):
            out[i] = col.reverse() if (flip and col is not None) else col
    if pp:
        for i, col in zip(pp_idx, batched_polygon_polygon(pp, geom, state)):
            out[i] = col
    return out


def build_contacts(pairs: List[Tuple[RigidBody, RigidBody]],
                   contacts: ContactSet = None,
                   state: Optional["transport.State"] = None) -> List[ContactConstraint]:
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
        When state is given (the B-bridge mirror), the block pose is asserted to
        equal the scalar bodies before any reader trusts it.
    """
    constraints = []
    candidates = [(a, b) for a, b in pairs if a.physics or b.physics]
    unique = {id(p): p for a, b in candidates for p in (a, b)}
    boxes = {bid: _broad_box(body, state) for bid, body in unique.items()}
    eligible = [(a, b) for a, b in candidates
                if not boxes[id(a)].disjoint(boxes[id(b)])]
    if state is not None:
        transport.assert_block_mirrors(state.block, state.row_of,
                                       [body for pair in eligible for body in pair])
    polys = list({p.uid: p for a, b in eligible for p in (a, b)
                  if isinstance(p, Polygon)}.values())
    geom = transport.GeometryPool(polys)
    if state is not None:
        geom.sync_from_block(state.block, state.row_of)
    resolved = _batch_circle_collisions(eligible, geom, state)
    hits = []
    pp_pairs = []
    for i, (a, b) in enumerate(eligible):
        collision = resolved[i] if i in resolved else detect_collision(a, b)
        if collision is None or collision.depth <= 0:
            continue
        grid_k = None
        if isinstance(a, Polygon) and isinstance(b, Polygon):
            grid_k = len(pp_pairs)
            pp_pairs.append((a, b))
        hits.append((a, b, collision, grid_k))
    manifolds = batched_contact_points(geom, pp_pairs, state=state) if pp_pairs else None
    for a, b, collision, grid_k in hits:
        normal = collision.normal
        idx_a = state.row_of.get(a.uid) if state is not None else None
        idx_b = state.row_of.get(b.uid) if state is not None else None
        if grid_k is not None:
            # Packed row: count, then stride-6 blocks [px, py, ra_x, ra_y, rb_x, rb_y] per point.
            k = grid_k
            points = []
            for i in range(int(manifolds[k, 0])):
                o = 1 + 6 * i
                points.append((manifolds[k, o], manifolds[k, o + 1],
                               manifolds[k, o + 2:o + 4], manifolds[k, o + 4:o + 6]))
        else:
            c0, c1, _id0, _id1 = find_contact_points(a, b, collision, geom, state)
            ca = transport.block_center(a, state)
            cb = transport.block_center(b, state)
            points = [(c.x, c.y, c - ca, c - cb)
                      for c in (c0, c1) if c is not None]
        for px, py, r_a, r_b in points:
            if contacts is not None:
                contacts.add((px, py))
            bias_velocity = relative_normal_velocity(a, b, r_a, r_b, normal)
            constraints.append(ContactConstraint(a, b, normal, r_a, r_b, collision.depth,
                                                 bias_velocity, idx_a, idx_b))
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
    for a, b, normal, r_a, r_b, depth, _bias, _ia, _ib in constraints:
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
        a, b, normal, r_a, r_b, _depth, bias_velocity, _ia, _ib = constraint
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
                  sub_dt: float, contacts: ContactSet = None,
                  state: Optional["transport.State"] = None):
    """Advance the dynamic bodies one XPBD sub-step: integrate, solve positions, derive, solve velocities."""
    previous = snapshot_poses(bodies)
    integrate_block(bodies, gravity, sub_dt)
    if state is not None:
        state.gather()
    constraints = build_contacts(pairs, contacts, state)
    lambdas = solve_positions(constraints)
    derive_velocities(bodies, previous, sub_dt)
    solve_velocities(physics, constraints, lambdas, sub_dt, gravity)


def solve_group_substep(physics: Physics, bodies: List[RigidBody],
                        pairs: List[Tuple[RigidBody, RigidBody]], gravity: Matrix,
                        sub_dt: float, num_substeps: int, contacts: ContactSet = None,
                        state: Optional["transport.State"] = None):
    """Advance one group of bodies over all sub-steps with the XPBD solver."""
    for _ in range(num_substeps):
        solve_substep(physics, bodies, pairs, gravity, sub_dt, contacts, state)
