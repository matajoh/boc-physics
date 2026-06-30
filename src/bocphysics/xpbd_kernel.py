"""Colour-batched SoA XPBD solver: the data-parallel core of xpbd.solve_substep.

Description:
    The serial XPBD path in xpbd.py corrects penetration and resolves friction /
    restitution one contact at a time. This module runs the identical solve as
    data-parallel Matrix kernels: the constraints are edge-coloured so that every
    constraint in a colour touches disjoint movable bodies, each colour is packed
    into per-contact (K x n) blocks, and one colour solves in a handful of
    row-wise ops with a scatter-add back onto the shared body blocks. Within a
    colour the bodies are disjoint, so the batched Jacobi update equals the serial
    sweep over that colour exactly; across colours the solve is Gauss-Seidel in
    colour order. That reordering is a valid re-linearisation gated by settling-
    band tests, not the bit-exact golden master -- the same contract the impulse
    batched kernel in kernel.py carries. Same physics as xpbd.py, different
    scheduler: the position magnitude depth / w is built once from fixed lever
    arms, so the position pass is order-independent, and the velocity pass scatters
    the friction impulse and then the restitution impulse separately so a body's
    two updates round exactly as the serial pair of apply_velocity_impulse calls.
"""

from typing import Dict, List, NamedTuple, Tuple

from bocpy import Matrix

from . import solver, xpbd
from .bodies import RigidBody
from .kernel import greedy_edge_color, pack_bodies
from .physics import Physics

INF = float("inf")


class ConstraintBlock(NamedTuple):
    """One colour's contacts packed into SoA blocks plus the body row indices."""

    normal: Matrix
    r_a: Matrix
    r_b: Matrix
    ra_perp: Matrix
    rb_perp: Matrix
    depth: Matrix
    bias: Matrix
    idx_a: List[int]
    idx_b: List[int]


def _endpoint_id(body: RigidBody, fresh: List[int]) -> int:
    """Return a movable body's uid, or a fresh unique negative id for a static body."""
    if body.physics:
        return body.uid

    fresh[0] -= 1
    return fresh[0]


def colour_contacts(constraints: List[xpbd.ContactConstraint]) -> List[list]:
    """Partition constraints into colours of mutually body-disjoint contacts.

    Description:
        Two constraints conflict only when they share a movable body, so a
        dynamic body is its own uid endpoint while each static occurrence gets a
        fresh unique negative id that can never collide. Greedy edge-colouring in
        input order is worker-count independent, so the colouring is deterministic.
    """
    fresh = [0]
    items = [(_endpoint_id(c.a, fresh), _endpoint_id(c.b, fresh)) for c in constraints]
    colours, ncolours = greedy_edge_color(items)
    groups: List[list] = [[] for _ in range(ncolours)]
    for constraint, colour in zip(constraints, colours):
        groups[colour].append(constraint)

    return groups


def constraint_body_rows(constraints: List[xpbd.ContactConstraint]) -> Tuple[list, Dict[int, int]]:
    """Collect the bodies the constraints touch and map each to a SoA row index."""
    rows: Dict[int, int] = {}
    bodies: list = []
    for constraint in constraints:
        for body in (constraint.a, constraint.b):
            if id(body) not in rows:
                rows[id(body)] = len(bodies)
                bodies.append(body)

    return bodies, rows


def pack_poses(bodies: list) -> Tuple[Matrix, Matrix, Matrix, Matrix]:
    """Stack per-body position, angle, and inverse mass / inertia into SoA blocks."""
    n = len(bodies)
    position = Matrix(n, 2, [c for b in bodies for c in (b.position.x, b.position.y)])
    angle = Matrix(n, 1, [b.angle for b in bodies])
    inv_m = Matrix(n, 1, [b.inv_mass for b in bodies])
    inv_i = Matrix(n, 1, [b.inv_inertia for b in bodies])
    return position, angle, inv_m, inv_i


def pack_colour(colour: list, rows: Dict[int, int]) -> ConstraintBlock:
    """Flatten one colour's constraints into per-contact SoA blocks and row indices."""
    n_rows, ra_rows, rb_rows, rap_rows, rbp_rows = [], [], [], [], []
    depth_rows, bias_rows, idx_a, idx_b = [], [], [], []
    for a, b, normal, r_a, r_b, depth, bias, _ia, _ib in colour:
        ra_perp = r_a.perpendicular()
        rb_perp = r_b.perpendicular()
        n_rows += [normal.x, normal.y]
        ra_rows += [r_a.x, r_a.y]
        rb_rows += [r_b.x, r_b.y]
        rap_rows += [ra_perp.x, ra_perp.y]
        rbp_rows += [rb_perp.x, rb_perp.y]
        depth_rows.append(depth)
        bias_rows.append(bias)
        idx_a.append(rows[id(a)])
        idx_b.append(rows[id(b)])

    k = len(depth_rows)
    return ConstraintBlock(Matrix(k, 2, n_rows), Matrix(k, 2, ra_rows), Matrix(k, 2, rb_rows),
                           Matrix(k, 2, rap_rows), Matrix(k, 2, rbp_rows),
                           Matrix(k, 1, depth_rows), Matrix(k, 1, bias_rows), idx_a, idx_b)


def position_kernel(pos: Matrix, ang: Matrix, inv_m: Matrix, inv_i: Matrix,
                    block: ConstraintBlock) -> Matrix:
    """One batched position pass over a colour; scatter the pose deltas, return the normal lambda.

    Description:
        Each contact is pushed apart by depth / w along its normal, where w is the
        summed generalised inverse mass (compliance alpha = 0). The lever arms are
        fixed, so the magnitude is order-independent; disjoint bodies in the colour
        make the scatter-add equal to the serial apply_positional_impulse sweep.
        The returned lambda feeds the friction bound in velocity_kernel.
    """
    ia, ib = block.idx_a, block.idx_b
    ima, imb = inv_m[ia], inv_m[ib]
    iia, iib = inv_i[ia], inv_i[ib]
    ra_x_n = block.r_a.cross(block.normal, axis=1)
    rb_x_n = block.r_b.cross(block.normal, axis=1)
    w = ima + ra_x_n * ra_x_n * iia + imb + rb_x_n * rb_x_n * iib
    lam_n = block.depth / w
    impulse = lam_n * block.normal
    pos.put(ia, impulse * (ima * -1.0), accumulate=True)
    pos.put(ib, impulse * imb, accumulate=True)
    ang.put(ia, block.r_a.cross(impulse, axis=1) * (iia * -1.0), accumulate=True)
    ang.put(ib, block.r_b.cross(impulse, axis=1) * iib, accumulate=True)
    return lam_n


def velocity_kernel(physics: Physics, vel: Matrix, spin: Matrix, inv_m: Matrix,
                    inv_i: Matrix, block: ConstraintBlock, lam_n: Matrix, h: float,
                    g: float):
    """One batched velocity pass over a colour: dynamic Coulomb friction then restitution.

    Description:
        Friction caps the tangential change at the Coulomb bound mu_d * f_n with
        f_n = lambda_n / h^2 from the position pass; restitution adds back
        -e * bias along the normal, gated off (e = 0) when the approach speed is at
        the gravity scale 2 * g * h. Both impulses are computed from the same
        pre-solve velocity, then scattered separately -- friction first, then
        restitution -- so a body's two updates round exactly as the serial pair of
        apply_velocity_impulse calls. Disjoint bodies make the colour bit-exact
        with the serial sweep over the same contacts.
    """
    ia, ib = block.idx_a, block.idx_b
    ima, imb = inv_m[ia], inv_m[ib]
    iia, iib = inv_i[ia], inv_i[ib]
    normal = block.normal
    rel = vel[ib]
    rel.scaled_add(spin[ib], block.rb_perp, in_place=True)
    contact_a = vel[ia]
    contact_a.scaled_add(spin[ia], block.ra_perp, in_place=True)
    rel.subtract(contact_a, out=rel)
    vn = rel.vecdot(normal, axis=1)
    rel.scaled_add(vn * -1.0, normal, in_place=True)  # rel now holds the tangential component vt
    vt_mag = rel.magnitude_squared(axis=1).sqrt()

    live = Matrix.greater(vt_mag, xpbd.EPS)
    vt_safe = Matrix.where(live, vt_mag, Matrix(vt_mag.rows, 1, [1.0] * vt_mag.rows))
    tangent = rel / vt_safe
    ra_x_t = block.r_a.cross(tangent, axis=1)
    rb_x_t = block.r_b.cross(tangent, axis=1)
    w_t = ima + ra_x_t * ra_x_t * iia + imb + rb_x_t * rb_x_t * iib
    f_n = lam_n / (h * h)
    bound = (h * physics.dynamic_friction) * f_n
    dvt = Matrix.where(Matrix.less(bound, vt_mag), bound, vt_mag) * -1.0
    wt_live = Matrix.greater(w_t, xpbd.EPS)
    wt_safe = Matrix.where(wt_live, w_t, Matrix(w_t.rows, 1, [1.0] * w_t.rows))
    friction = tangent * (dvt / wt_safe * live * wt_live)
    _scatter_impulse(vel, spin, block, ima, imb, iia, iib, friction)

    keep = Matrix.greater(vn.abs(), 2.0 * g * h)
    e = keep * physics.restitution
    rebound = (e * block.bias * -1.0).clip(0.0, INF)
    w_n = ima + ra_x_n_sq(block.r_a, normal) * iia + imb + ra_x_n_sq(block.r_b, normal) * iib
    dvn = vn * -1.0 + rebound
    restitution = normal * (dvn / w_n)
    _scatter_impulse(vel, spin, block, ima, imb, iia, iib, restitution)


def ra_x_n_sq(r: Matrix, normal: Matrix) -> Matrix:
    """Square of the row-wise lever cross-normal, the rotational term of inverse mass along normal."""
    rn = r.cross(normal, axis=1)
    return rn * rn


def _scatter_impulse(vel: Matrix, spin: Matrix, block: ConstraintBlock, ima: Matrix,
                     imb: Matrix, iia: Matrix, iib: Matrix, impulse: Matrix):
    """Scatter-add one velocity impulse onto both bodies: a gets -impulse, b gets +impulse."""
    ia, ib = block.idx_a, block.idx_b
    vel.put(ia, impulse * (ima * -1.0), accumulate=True)
    vel.put(ib, impulse * imb, accumulate=True)
    spin.put(ia, block.r_a.cross(impulse, axis=1) * (iia * -1.0), accumulate=True)
    spin.put(ib, block.r_b.cross(impulse, axis=1) * iib, accumulate=True)


def solve_substep(physics: Physics, bodies: List[RigidBody],
                  pairs: List[Tuple[RigidBody, RigidBody]], gravity: Matrix,
                  sub_dt: float, contacts: xpbd.ContactSet = None):
    """Advance the dynamic bodies one XPBD sub-step with the colour-batched kernels."""
    previous = xpbd.snapshot_poses(bodies)
    solver.integrate_block(bodies, gravity, sub_dt)
    constraints = xpbd.build_contacts(pairs, contacts)
    if constraints:
        touched, rows = constraint_body_rows(constraints)
        pos, ang, inv_m, inv_i = pack_poses(touched)
        colours = [pack_colour(colour, rows) for colour in colour_contacts(constraints)]
        lambdas = [position_kernel(pos, ang, inv_m, inv_i, block) for block in colours]
        for body in touched:
            if body.physics:
                i = rows[id(body)]
                body.position = pos[i]
                body.angle = ang[i, 0]
                body.update_needed_ = True

    xpbd.derive_velocities(bodies, previous, sub_dt)
    if constraints:
        vel, spin, _inv_m, _inv_i = pack_bodies(touched)
        g = gravity.magnitude()
        for block, lam_n in zip(colours, lambdas):
            velocity_kernel(physics, vel, spin, inv_m, inv_i, block, lam_n, sub_dt, g)
        for body in touched:
            if body.physics:
                i = rows[id(body)]
                body.linear_velocity = vel[i]
                body.angular_velocity = spin[i, 0]


def solve_group_substep(physics: Physics, bodies: List[RigidBody],
                        pairs: List[Tuple[RigidBody, RigidBody]], gravity: Matrix,
                        sub_dt: float, num_substeps: int, contacts: xpbd.ContactSet = None):
    """Advance one group of bodies over all sub-steps with the batched XPBD solver."""
    for _ in range(num_substeps):
        solve_substep(physics, bodies, pairs, gravity, sub_dt, contacts)
