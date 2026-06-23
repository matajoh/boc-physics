"""Colour-batched SoA velocity solver: the vectorised core of resolve_pair_list.

Description:
    The per-contact velocity solver in solver.resolve_pair_list applies each
    manifold's impulse one Python call at a time. This module solves the same
    velocity iteration as data-parallel Matrix kernels: the manifolds are edge-
    coloured so that every manifold in a colour touches disjoint movable bodies,
    each colour is packed into per-contact (K x n) blocks, and one colour solves
    in a handful of row-wise ops with a scatter-add back onto the shared body
    velocity blocks. Within a colour the bodies are disjoint, so the batched
    Jacobi update equals the stock sequential sweep over that colour exactly;
    across colours the solve is Gauss-Seidel, identical in structure to the
    per-contact loop but visiting manifolds in colour order rather than pair
    order. That reordering changes the finite-iteration result like any valid
    re-linearisation, so this path is gated by settling-band tests, not by the
    bit-exact golden master.

    Like resolve_pair_list, this kernel runs accumulated PGS for FRICTION: each
    colour carries a per-contact running normal and tangent impulse (lambda)
    across the sweeps, clamps the running total rather than the increment, and
    applies only the change, so a later sweep can release earlier over-push.
    ROTATION mode keeps the stock non-accumulated normal sweep, mirroring the
    serial path's per-mode split. The colours visit in a fixed body-disjoint
    order rather than the serial path's gravity-aligned apex-first order, so for
    FRICTION the two are not bit-identical; that path stays gated by settling-
    band tests, not the bit-exact golden master.
"""

import math
from typing import Dict, List, Tuple

from bocpy import Matrix

from .config import PhysicsMode
from .physics import Physics

INF = math.inf


def greedy_edge_color(items: List[Tuple[int, int]]) -> Tuple[List[int], int]:
    """Greedily colour edges so two sharing an endpoint never share a colour.

    Description:
        Each item is an (endpoint_a, endpoint_b) pair; this is greedy colouring
        of the line graph in input order -- assign each item the smallest colour
        not already used by an item incident to either endpoint. Input order is
        deterministic, so the colouring is worker-count independent.
    """
    used: Dict[int, set] = {}
    colors = [0] * len(items)
    ncolors = 0
    for idx, (end_a, end_b) in enumerate(items):
        taken = used.get(end_a, set()) | used.get(end_b, set())
        color = 0
        while color in taken:
            color += 1

        colors[idx] = color
        used.setdefault(end_a, set()).add(color)
        used.setdefault(end_b, set()).add(color)
        ncolors = max(ncolors, color + 1)

    return colors, ncolors


def colour_manifolds(manifolds: list) -> List[list]:
    """Partition manifolds into colours of mutually body-disjoint constraints.

    Description:
        A constraint conflicts with another only when they share a movable body,
        so a dynamic body is its own uid endpoint while each static occurrence
        gets a fresh unique negative id that can never collide. The result is a
        list of colour groups, each a list of manifolds whose movable bodies are
        pairwise disjoint -- the unit a single batched colour solve consumes.
    """
    fresh = [0]
    items = []
    for a, b, _collision, _c0, _c1 in manifolds:
        items.append((_endpoint_id(a, fresh), _endpoint_id(b, fresh)))

    colors, ncolors = greedy_edge_color(items)
    groups: List[list] = [[] for _ in range(ncolors)]
    for manifold, color in zip(manifolds, colors):
        groups[color].append(manifold)

    return groups


def _endpoint_id(body, fresh: List[int]) -> int:
    """Return a movable body's uid, or a fresh unique id for a static body."""
    if body.physics:
        return body.uid

    fresh[0] -= 1
    return fresh[0]


def body_rows(manifolds: list) -> Tuple[list, Dict[int, int]]:
    """Collect the bodies the manifolds touch and map each to a SoA row index.

    Description:
        Walks the manifolds in order, assigning each distinct body (keyed by
        identity) the next row index. Returns the body list in row order and the
        identity -> row map the packers and scatter use.
    """
    rows: Dict[int, int] = {}
    bodies = []
    for a, b, _collision, _c0, _c1 in manifolds:
        for body in (a, b):
            if id(body) not in rows:
                rows[id(body)] = len(bodies)
                bodies.append(body)

    return bodies, rows


def pack_bodies(bodies: list):
    """Stack per-body velocity, spin, and inverse mass / inertia into SoA blocks.

    Description:
        Static bodies get a zeroed velocity row to match the stock solver's
        ZERO_VEC read, and their inverse mass / inertia are already zero so any
        scatter delta onto their row is exactly zero.
    """
    n = len(bodies)
    vel = Matrix(n, 2, [c for b in bodies for c in
                        ((b.linear_velocity.x, b.linear_velocity.y)
                         if b.physics else (0.0, 0.0))])
    spin = Matrix(n, 1, [b.angular_velocity if b.physics else 0.0
                         for b in bodies])
    inv_m = Matrix(n, 1, [b.inv_mass for b in bodies])
    inv_i = Matrix(n, 1, [b.inv_inertia for b in bodies])
    return vel, spin, inv_m, inv_i


def pack_contacts(physics: Physics, manifolds: list, rows: Dict[int, int]):
    """Flatten one colour's manifolds into per-contact SoA arrays.

    Description:
        Each contact contributes a normal, a fixed tangent axis and its tangent
        effective mass, prepared lever arms (ra, rb and their perpendiculars), an
        effective-mass denominator, the body row indices, and a 1 / num_contacts
        weight. The prepared data comes from the same Physics.prepare_contacts the
        serial solver uses, and the tangent axis / mass match build_tangent_data,
        so the per-contact maths is identical -- only the evaluation is batched.
    """
    n_rows, ra_rows, rb_rows = [], [], []
    rap_rows, rbp_rows, denom_rows, weight_rows = [], [], [], []
    t_rows, denomt_rows, vtarget_rows = [], [], []
    idx_a, idx_b = [], []
    for a, b, collision, c0, c1 in manifolds:
        normal = collision.normal
        t_axis = normal.perpendicular()
        prepared = physics.prepare_contacts(a, b, normal, c0, c1)
        for contact in prepared:
            n_rows += [normal.x, normal.y]
            t_rows += [t_axis.x, t_axis.y]
            ra_rows += [contact.ra.x, contact.ra.y]
            rb_rows += [contact.rb.x, contact.rb.y]
            rap_rows += [contact.ra_perp.x, contact.ra_perp.y]
            rbp_rows += [contact.rb_perp.x, contact.rb_perp.y]
            ra_perp_dot_t = contact.ra_perp.vecdot(t_axis)
            rb_perp_dot_t = contact.rb_perp.vecdot(t_axis)
            denomt_rows.append(a.inv_mass + b.inv_mass +
                               ra_perp_dot_t * ra_perp_dot_t * a.inv_inertia +
                               rb_perp_dot_t * rb_perp_dot_t * b.inv_inertia)
            denom_rows.append(contact.denom)
            vtarget_rows.append(contact.v_target)
            weight_rows.append(1.0 / len(prepared))
            idx_a.append(rows[id(a)])
            idx_b.append(rows[id(b)])

    k = len(denom_rows)
    blocks = {"n": Matrix(k, 2, n_rows), "t": Matrix(k, 2, t_rows),
              "ra": Matrix(k, 2, ra_rows),
              "rb": Matrix(k, 2, rb_rows), "ra_perp": Matrix(k, 2, rap_rows),
              "rb_perp": Matrix(k, 2, rbp_rows),
              "denom": Matrix(k, 1, denom_rows),
              "denom_t": Matrix(k, 1, denomt_rows),
              "v_target": Matrix(k, 1, vtarget_rows),
              "weight": Matrix(k, 1, weight_rows)}
    return blocks, idx_a, idx_b


def scatter_deltas(vel, spin, idx_a, idx_b, dva, dvb, dwa, dwb):
    """Scatter-add per-contact velocity / spin deltas onto body rows.

    Description:
        put(accumulate=True) folds duplicate indices additively, so a 2-contact
        manifold writing both points onto one body sums correctly in a single
        scatter per side. Statics repeat across manifolds but carry zero deltas.
        Mutates vel and spin in place.
    """
    vel.put(idx_a, dva, accumulate=True)
    vel.put(idx_b, dvb, accumulate=True)
    spin.put(idx_a, dwa, accumulate=True)
    spin.put(idx_b, dwb, accumulate=True)


def normal_kernel(vel, spin, inv_m, inv_i, blocks, idx_a, idx_b):
    """Run one non-accumulated normal-impulse iteration over a colour batch.

    Description:
        The ROTATION-mode normal sweep: gathers both bodies' velocity per contact,
        solves the impulse that drives the normal velocity to the captured
        restitution target (clip(0, inf) keeps it push-only, reproducing the stock
        vn > target skip), and scatters the deltas back. Clamps the per-iteration
        increment, not a running total, matching the serial apply_rotation sweep.
        Mutates vel and spin.
    """
    n_block = blocks["n"]
    va = vel[idx_a]
    vb = vel[idx_b]
    wa = spin[idx_a]
    wb = spin[idx_b]
    rel = (vb + blocks["rb_perp"] * wb) - (va + blocks["ra_perp"] * wa)
    vn = rel.vecdot(n_block, axis=1)
    j = ((blocks["v_target"] - vn) / blocks["denom"] * blocks["weight"]).clip(0.0, INF)
    impulse = j * n_block
    dva = impulse * (inv_m[idx_a] * -1.0)
    dvb = impulse * inv_m[idx_b]
    dwa = blocks["ra"].cross(impulse, axis=1) * (inv_i[idx_a] * -1.0)
    dwb = blocks["rb"].cross(impulse, axis=1) * inv_i[idx_b]
    scatter_deltas(vel, spin, idx_a, idx_b, dva, dvb, dwa, dwb)


def normal_accumulate(vel, spin, inv_m, inv_i, blocks, idx_a, idx_b, lam_n):
    """Run one accumulated normal-impulse iteration over a colour batch.

    Description:
        The FRICTION-mode normal sweep: gathers both bodies' velocity per contact,
        computes the impulse increment that drives the normal velocity to the
        captured restitution target, adds it to the running total lam_n and clamps
        the TOTAL to [0, inf) (push-only, reproducing the stock skip), then applies
        only the change so a later sweep can release earlier over-push. Returns the
        new running total. Mutates vel and spin.
    """
    n_block = blocks["n"]
    va = vel[idx_a]
    vb = vel[idx_b]
    wa = spin[idx_a]
    wb = spin[idx_b]
    rel = (vb + blocks["rb_perp"] * wb) - (va + blocks["ra_perp"] * wa)
    vn = rel.vecdot(n_block, axis=1)
    delta = (blocks["v_target"] - vn) / blocks["denom"] * blocks["weight"]
    new_total = (lam_n + delta).clip(0.0, INF)
    impulse = (new_total - lam_n) * n_block
    dva = impulse * (inv_m[idx_a] * -1.0)
    dvb = impulse * inv_m[idx_b]
    dwa = blocks["ra"].cross(impulse, axis=1) * (inv_i[idx_a] * -1.0)
    dwb = blocks["rb"].cross(impulse, axis=1) * inv_i[idx_b]
    scatter_deltas(vel, spin, idx_a, idx_b, dva, dvb, dwa, dwb)
    return new_total


def friction_accumulate(vel, spin, inv_m, inv_i, blocks, idx_a, idx_b, lam_n,
                        lam_t, sf, df):
    """Run one accumulated Coulomb-friction iteration over a colour batch.

    Description:
        Reads the post-normal velocity along the fixed tangent axis, adds the
        increment to the running tangent total lam_t, then selects the cone branch
        with masks: inside the static cone (|total| <= sf*lam_n) keeps the total,
        outside slides and clamps the total to the kinetic cone (+/- df*lam_n).
        Applies only the change and returns the new running total. The axis is
        fixed, so there is no tiny-tangent guard. Mutates vel and spin.
    """
    t_block = blocks["t"]
    va = vel[idx_a]
    vb = vel[idx_b]
    wa = spin[idx_a]
    wb = spin[idx_b]
    rel = (vb + blocks["rb_perp"] * wb) - (va + blocks["ra_perp"] * wa)
    vt = rel.vecdot(t_block, axis=1)
    jt = vt * -1.0 / blocks["denom_t"] * blocks["weight"]
    new_total = lam_t + jt
    static_bound = lam_n * sf
    kinetic_bound = lam_n * df
    neg_kinetic = kinetic_bound * -1.0
    clamped = Matrix.where(Matrix.greater(new_total, kinetic_bound),
                           kinetic_bound, new_total)
    clamped = Matrix.where(Matrix.less(clamped, neg_kinetic), neg_kinetic, clamped)
    inside = Matrix.less_equal(new_total.abs(), static_bound)
    new_total = Matrix.where(inside, new_total, clamped)
    impulse = (new_total - lam_t) * t_block
    dva = impulse * (inv_m[idx_a] * -1.0)
    dvb = impulse * inv_m[idx_b]
    dwa = blocks["ra"].cross(impulse, axis=1) * (inv_i[idx_a] * -1.0)
    dwb = blocks["rb"].cross(impulse, axis=1) * inv_i[idx_b]
    scatter_deltas(vel, spin, idx_a, idx_b, dva, dvb, dwa, dwb)
    return new_total


def solve_colour(physics, vel, spin, inv_m, inv_i, blocks, idx_a, idx_b, lam):
    """Solve one colour batch, mirroring resolve_pair_list's per-mode split.

    Description:
        Disjoint movable bodies make the all-normals-then-all-frictions order
        equal to the stock per-manifold normal-then-friction sweep, since each
        manifold's friction reads only its own post-normal velocity. FRICTION runs
        accumulated PGS -- lam is the colour's running [lam_n, lam_t] impulse pair,
        updated in place across sweeps -- while ROTATION runs the non-accumulated
        normal sweep and leaves lam untouched. Mutates vel, spin, and lam.
    """
    if physics.mode == PhysicsMode.FRICTION:
        lam[0] = normal_accumulate(vel, spin, inv_m, inv_i, blocks, idx_a, idx_b,
                                   lam[0])
        lam[1] = friction_accumulate(vel, spin, inv_m, inv_i, blocks, idx_a, idx_b,
                                     lam[0], lam[1], physics.static_friction,
                                     physics.dynamic_friction)
    else:
        normal_kernel(vel, spin, inv_m, inv_i, blocks, idx_a, idx_b)


def resolve_batched(physics: Physics, manifolds: list,
                    num_velocity_iterations: int):
    """Iterate the colour-batched velocity solve over a manifold set.

    Description:
        Collects the touched bodies into SoA velocity / spin blocks, colours the
        manifolds into body-disjoint batches, packs each colour's contacts once,
        seeds a per-colour running [lam_n, lam_t] impulse pair, then runs
        num_velocity_iterations Gauss-Seidel sweeps over the colours (each colour
        batched internally). FRICTION accumulates its lambda across sweeps;
        ROTATION leaves the seeded pair untouched. The new velocities are
        scattered back onto the bodies at the end. Equivalent in structure to the
        serial per-contact loop but visiting manifolds in colour order, so it
        converges close to the serial solution while differing at finite counts.
    """
    if not manifolds:
        return

    bodies, rows = body_rows(manifolds)
    vel, spin, inv_m, inv_i = pack_bodies(bodies)
    colours = colour_manifolds(manifolds)
    packed = [pack_contacts(physics, colour, rows) for colour in colours]
    lambdas = [[Matrix.zeros((blocks["denom"].rows, 1)),
                Matrix.zeros((blocks["denom"].rows, 1))]
               for blocks, _idx_a, _idx_b in packed]
    for _ in range(num_velocity_iterations):
        for (blocks, idx_a, idx_b), lam in zip(packed, lambdas):
            solve_colour(physics, vel, spin, inv_m, inv_i, blocks, idx_a, idx_b,
                         lam)

    for body in bodies:
        if body.physics:
            i = rows[id(body)]
            body.linear_velocity = vel[i]
            body.angular_velocity = spin[i, 0]
