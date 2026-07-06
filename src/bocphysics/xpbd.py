"""Serial 2D XPBD contact-solve primitives: position-based rigid-body dynamics.

Description:
    A faithful 2D specialisation of Mueller et al. 2020, "Detailed Rigid Body
    Simulation with Extended Position Based Dynamics" (Algorithm 2). Orientation
    is a scalar angle, inertia a scalar, and the cross products r x n / r x p
    collapse to scalars. This module holds the shared building blocks a sub-step
    is made of: re-evaluate the narrow phase at the new pose (build_contacts),
    run a single position pass (solve_positions, numPosIters = 1, compliance
    alpha = 0 for rigid contacts), derive the velocity from the position delta,
    then apply one velocity pass for dynamic Coulomb friction (Eqn 30) and
    restitution (Eqn 34). The jacobi module composes these into a full sub-step.
    The integrator is shared with the impulse path via solver.integrate_block;
    the import is strictly one-way (solver never imports this module), so no
    cycle is formed.
"""

import math
from typing import List, NamedTuple, Optional, Set, Tuple

from bocpy import Matrix

from .bodies import AABB, Circle, Polygon, RigidBody
from .collisions import (batched_circle_circle, batched_circle_polygon,
                         batched_polygon_polygon, detect_collision)
from .contacts import batched_contact_points, find_contact_points
from .physics import Physics

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


def _broad_box(body: RigidBody) -> AABB:
    """Conservative bounding-circle AABB from the body's scalar pose.

    Description:
        The broad-phase cull only needs a box that never shrinks below the true
        bounds, so the rotation-invariant bounding circle (centre +/- radius)
        suffices and needs no angle. The looser box only widens the cull -- it
        never rejects a real overlap -- so the emitted constraint set and its
        order are unchanged.
    """
    x, y = body.position.x, body.position.y
    rad = body.radius
    return AABB(x - rad, y - rad, x + rad, y + rad)


def relative_normal_velocity(a: RigidBody, b: RigidBody, r_a: Matrix,
                             r_b: Matrix, normal: Matrix) -> float:
    """Pre-solve relative velocity along the contact normal (b relative to a)."""
    return (contact_velocity(b, r_b) - contact_velocity(a, r_a)).vecdot(normal)


def _batch_circle_collisions(pairs, geom):
    """Resolve circle-circle and circle-poly pairs in two batched SAT calls.

    Returns a dict mapping each eligible pair's index to its Collision-or-None,
    in the same orientation detect_collision would yield. geom is the shared
    GeometryPool over every eligible polygon, reused by both batched SAT calls.
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
    for i, col in zip(cc_idx, batched_circle_circle(cc)) if cc else ():
        out[i] = col
    if cp:
        for i, flip, col in zip(cp_idx, cp_flip, batched_circle_polygon(cp, geom)):
            out[i] = col.reverse() if (flip and col is not None) else col
    if pp:
        for i, col in zip(pp_idx, batched_polygon_polygon(pp, geom)):
            out[i] = col
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
    candidates = [(a, b) for a, b in pairs if a.physics or b.physics]
    unique = {id(p): p for a, b in candidates for p in (a, b)}
    boxes = {bid: _broad_box(body) for bid, body in unique.items()}
    eligible = [(a, b) for a, b in candidates
                if not boxes[id(a)].disjoint(boxes[id(b)])]
    polys = list({p.uid: p for a, b in eligible for p in (a, b)
                  if isinstance(p, Polygon)}.values())
    geom = GeometryPool(polys)
    resolved = _batch_circle_collisions(eligible, geom)
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
    manifolds = batched_contact_points(geom, pp_pairs) if pp_pairs else None
    for a, b, collision, grid_k in hits:
        normal = collision.normal
        if grid_k is not None:
            # Packed row: count, then stride-6 blocks [px, py, ra_x, ra_y, rb_x, rb_y] per point.
            k = grid_k
            points = []
            for i in range(int(manifolds[k, 0])):
                o = 1 + 6 * i
                points.append((manifolds[k, o], manifolds[k, o + 1],
                               manifolds[k, o + 2:o + 4], manifolds[k, o + 4:o + 6]))
        else:
            c0, c1, _id0, _id1 = find_contact_points(a, b, collision, geom)
            ca = a.position
            cb = b.position
            points = [(c.x, c.y, c - ca, c - cb)
                      for c in (c0, c1) if c is not None]
        for px, py, r_a, r_b in points:
            if contacts is not None:
                contacts.add((px, py))
            bias_velocity = relative_normal_velocity(a, b, r_a, r_b, normal)
            constraints.append(ContactConstraint(a, b, normal, r_a, r_b, collision.depth,
                                                 bias_velocity))
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


def _rotate(v: Matrix, angle: float) -> Matrix:
    """Rotate a 2D vector CCW by angle (radians) via the perpendicular basis.

    Description:
        R(a) v = v cos a + perp(v) sin a, so this needs only scalar-multiply,
        add, and perpendicular -- the ops a contact lever arm supports whether it
        is a Matrix.vector (circle) or a manifold row slice (polygon). It never
        touches v.x / v.y, which a slice does not expose.
    """
    return v * math.cos(angle) + v.perpendicular() * math.sin(angle)


def _contact_point_slip(body: RigidBody, r: Matrix, prev_pose: dict) -> Matrix:
    """World displacement of the contact's material point on body over the sub-step.

    Description:
        The point rides rigidly with the body, so its previous world position is
        the previous centre plus the lever arm rotated back by the pose change. A
        static body (absent from prev_pose) never moves, so its point does not
        slip. Returns current minus previous position, in world space.
    """
    prev = prev_pose.get(id(body))
    if prev is None:
        return _ZERO_VELOCITY
    prev_center, prev_angle = prev
    return (body.position + r) - (prev_center + _rotate(r, prev_angle - body.angle))


def _apply_static_friction(a: RigidBody, b: RigidBody, normal: Matrix, r_a: Matrix,
                           r_b: Matrix, lambda_n: float, mu_s: float, prev_pose: dict):
    """Cancel the contact's tangential slip while it stays inside the static cone.

    Description:
        Mueller et al. 2020 Algorithm 2 static friction, applied at the position
        level: the tangential drift of the two coincident material points since
        the sub-step start is removed outright while the tangential lambda stays
        within mu_s * lambda_n (the contact sticks); past the cone it is left to
        the velocity pass to slide at the dynamic bound. Cancelling the slip in
        position drives the derived velocity tangentially to zero, so a resting
        stack holds instead of creeping outward frame after frame.
    """
    slip = _contact_point_slip(a, r_a, prev_pose) - _contact_point_slip(b, r_b, prev_pose)
    slip_t = slip - normal * slip.vecdot(normal)
    mag = math.sqrt(slip_t.vecdot(slip_t))
    if mag < EPS:
        return
    t = slip_t / mag
    w_t = generalized_inverse_mass(a, r_a, t) + generalized_inverse_mass(b, r_b, t)
    if w_t < EPS:
        return
    if mag / w_t <= mu_s * lambda_n:
        apply_positional_impulse(a, b, r_a, r_b, slip_t / w_t)


def solve_positions(constraints: List[ContactConstraint], physics: Optional[Physics] = None,
                    prev_pose: Optional[dict] = None) -> List[float]:
    """One Gauss-Seidel position pass; return the normal lambda per constraint, in order.

    Description:
        Each penetrating contact is pushed apart along its normal by depth / w,
        where w is the summed generalised inverse mass (compliance alpha = 0).
        The returned lambda feeds the friction bound in solve_velocities; the
        list is one-to-one with constraints so the velocity pass can zip them.
        When prev_pose is given, static friction (Algorithm 2) is applied right
        after each normal push, sticking resting contacts so stacks hold; when it
        is omitted there is no prior pose to measure the slip against.
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
        if prev_pose is not None:
            _apply_static_friction(a, b, normal, r_a, r_b, magnitude,
                                   physics.static_friction, prev_pose)
    return lambdas


def snapshot_poses(bodies: List[RigidBody]) -> List[Tuple[Matrix, float]]:
    """Record each body's pose as (position copy, angle) so derive_velocities reads no aliased Matrix."""
    return [(body.position.copy(), body.angle) for body in bodies]


def derive_velocities(bodies: List[RigidBody],
                      previous: List[Tuple[Matrix, float]], h: float):
    """Set each body's velocity from its position delta over the sub-step (the XPBD velocity update)."""
    for body, (prev_pos, pa) in zip(bodies, previous):
        body.linear_velocity = (body.position - prev_pos) / h
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


class GeometryPool:
    """Transformed polygon geometry as padded SoA rows for batched SAT.

    Description:
        One row per polygon, statics included; circles have no geometry and are
        absent. Local vertices/normals are stored padded once at rebuild; sync
        rotates+translates the whole base block in place -- one batched column
        rotation, no per-poly Python loop -- so geom_x/geom_y/norm_x/norm_y hold
        the current world pose. The contact batchers read rows by uid via take.
    """

    def __init__(self, bodies: List[RigidBody]):
        """Build the geometry rows and uid->row map from the polygons."""
        self.rebuild(bodies)

    def rebuild(self, bodies: List[RigidBody]):
        """Reseed padded base rows and row_of after a body-set change; sync once."""
        self.polys = [b for b in bodies if isinstance(b, Polygon)]
        self.row_of = {b.uid: i for i, b in enumerate(self.polys)}
        self.vcount = {}
        if not self.polys:
            self.vmax = self.nmax = 0
            self.geom_x = self.geom_y = self.norm_x = self.norm_y = None
            return

        vmax = max(len(p.vertices) for p in self.polys)
        nmax = max(len(p.normals) for p in self.polys)
        self.vmax = vmax
        self.nmax = nmax
        rows = len(self.polys)
        vx = [0] * rows * vmax
        vy = [0] * rows * vmax
        nx = [0] * rows * nmax
        ny = [0] * rows * nmax
        vscan = 0
        nscan = 0
        for p in self.polys:
            self.vcount[p.uid] = len(p.vertices)
            for i, v in enumerate(p.vertices):
                vx[vscan + i] = v.x
                vy[vscan + i] = v.y
            for i in range(len(p.vertices), vmax):
                vx[vscan + i] = p.vertices[0].x
                vy[vscan + i] = p.vertices[0].y

            for i, n in enumerate(p.normals):
                nx[nscan + i] = n.x
                ny[nscan + i] = n.y

            vscan += vmax
            nscan += nmax

        self.base_vx = Matrix(rows, vmax, vx)
        self.base_vy = Matrix(rows, vmax, vy)
        self.base_nx = Matrix(rows, nmax, nx)
        self.base_ny = Matrix(rows, nmax, ny)
        self.px = [0] * rows
        self.py = [0] * rows
        self.cos = [0] * rows
        self.sin = [0] * rows
        self.sync()

    def sync(self):
        """Refresh world pose from the polygons' current scalar transforms."""
        for i, p in enumerate(self.polys):
            self.cos[i] = math.cos(p.angle)
            self.sin[i] = math.sin(p.angle)
            self.px[i] = p.position.x
            self.py[i] = p.position.y
        self._apply_pose()

    def _apply_pose(self):
        """Rotate+translate the base block into world pose, one batched pass."""
        rows = len(self.polys)
        if rows == 0:
            return
        cos = Matrix(rows, 1, self.cos)
        sin = Matrix(rows, 1, self.sin)
        px = Matrix(rows, 1, self.px)
        py = Matrix(rows, 1, self.py)

        self.geom_x = self.base_vx * cos
        self.geom_x -= self.base_vy * sin
        self.geom_x += px

        self.geom_y = self.base_vx * sin
        self.geom_y += self.base_vy * cos
        self.geom_y += py

        self.norm_x = self.base_nx * cos
        self.norm_x -= self.base_ny * sin

        self.norm_y = self.base_nx * sin
        self.norm_y += self.base_ny * cos

    def world_vertices(self, uid) -> Matrix:
        """Return one polygon's transformed vertices as a real-count (nv x 2) block."""
        r = self.row_of[uid]
        nv = self.vcount[uid]
        return Matrix.concat([self.geom_x[r, :nv].T, self.geom_y[r, :nv].T], 1)
