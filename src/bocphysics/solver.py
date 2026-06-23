"""Shared substep solver core, callable by the serial engine and parallel workers.

Description:
    These are module-level free functions over plain bodies, pairs, and a
    physics system, so a worker sub-interpreter can run the exact same solve as
    the serial engine -- "same core, different scheduler". No engine instance is
    required, which is what lets the parallel path reuse this verbatim.
"""

from typing import List, Optional, Set, Tuple

from bocpy import Matrix

from .bodies import RigidBody
from .collisions import detect_collision
from .config import PhysicsMode
from .contacts import find_contact_points, separate
from .kernel import resolve_batched
from .physics import Physics, TangentData


Manifold = Tuple
ContactSet = Optional[Set[Tuple[float, float]]]

# A/B toggle: when True, ROTATION/FRICTION velocity solves use the batched kernel
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

    velocity = velocity + gravity * dt
    position = position + velocity * dt
    angle = angle + spin * dt

    for i, body in enumerate(bodies):
        body.linear_velocity = velocity[i]
        body.position = position[i]
        body.angle = angle[i, 0]
        body.update_needed_ = True


def build_manifold(a: RigidBody, b: RigidBody, contacts: ContactSet):
    """Run the narrow phase for one pair, returning its contact manifold.

    Description:
        Confirms the exact collision and generates contact points, the
        geometry-only half of the narrow phase. Returns None for a false
        positive. When contacts is not None, the contact points are recorded
        into it for the show-contacts overlay. This is a pure query: it does
        not move the bodies -- positional correction (separate_manifold) is the
        caller's responsibility, so building is order-independent.
    """
    collision = detect_collision(a, b)
    if collision is None:
        # false positive
        return None

    c0, c1, _id0, _id1 = find_contact_points(a, b, collision)
    if contacts is not None:
        contacts.add((c0.x, c0.y))
        if c1 is not None:
            contacts.add((c1.x, c1.y))

    return (a, b, collision, c0, c1)


def separate_manifold(manifold: Manifold):
    """Apply positional penetration recovery for one built manifold.

    Description:
        Pushes the manifold's two bodies apart along the contact normal, the
        positional-correction half of contact handling kept out of build_manifold
        so the narrow phase stays a pure, order-independent geometry query. The
        depth is re-measured at correction time so a sequence of corrections is
        Gauss-Seidel (each sees its predecessors' moves), matching the engine's
        prior settling without making the manifold set order-dependent.
    """
    a, b, _collision, _c0, _c1 = manifold
    current = detect_collision(a, b)
    if current is not None:
        separate(a, b, current)


def build_group_manifolds(pairs: List[Tuple[RigidBody, RigidBody]],
                          contacts: ContactSet) -> List[Manifold]:
    """Build each pair's manifold once per sub-step, dropping false positives.

    Description:
        Pairs where neither body is dynamic are skipped: a static-static contact
        moves nothing and feeds a zero effective mass (denom == 0) into the
        velocity solve, which would divide by zero. Both callers already drop
        these upstream, so this is a self-protecting guard on the shared core's
        denom > 0 invariant, not a behavior change for the current inputs.
    """
    manifolds = []
    for a, b in pairs:
        if not (a.physics or b.physics):
            continue

        manifold = build_manifold(a, b, contacts)
        if manifold is not None:
            manifolds.append(manifold)

    return manifolds


def constraint_height(constraint) -> float:
    """Mean y of a constraint's two bodies; larger y is lower under +y gravity.

    Description:
        The ordering key for the gravity-aligned top-down sweep: sorting
        constraints ascending visits the apex (smallest y) first, so load
        propagates down the stack the way gravity drives it. Ties are broken by
        the stable sort preserving the deterministic manifold build order; do NOT
        add a uid tie-break -- uid is None on un-added bodies.
    """
    return (constraint.a.position.y + constraint.b.position.y) * 0.5


def build_tangent_data(constraint) -> TangentData:
    """Hoist a constraint's fixed friction axis and per-contact effective mass.

    Description:
        The pose is frozen for the whole velocity loop, so the tangent axis
        (normal.perpendicular()) and each contact's tangent effective mass are
        constant across sweeps. Computing them once per constraint here -- not per
        sweep inside apply_accumulated -- is a pure hoist, bit-identical to the
        validated probe which built the same expressions in the same order.
    """
    a = constraint.a
    b = constraint.b
    t_axis = constraint.normal.perpendicular()
    denom_t = []
    for contact in constraint.contacts:
        ra_perp_dot_t = contact.ra_perp.vecdot(t_axis)
        rb_perp_dot_t = contact.rb_perp.vecdot(t_axis)
        denom_t.append(a.inv_mass + b.inv_mass +
                       (ra_perp_dot_t * ra_perp_dot_t) * a.inv_inertia +
                       (rb_perp_dot_t * rb_perp_dot_t) * b.inv_inertia)

    return TangentData(t_axis, tuple(denom_t))


def resolve_pair_list(physics: Physics, manifolds: List[Manifold],
                      num_velocity_iterations: int, batched=None):
    """Prepare each manifold once, then iterate the cheap velocity solve.

    Description:
        The bodies' poses are fixed for the whole velocity loop, so each pair's
        lever arms and effective masses are constant. Preparing them once and
        iterating only the velocity-dependent work is a pure hoist -- bit-exact
        with resolving each manifold from scratch every iteration, just cheaper.
        When the batched flag is set and the mode is a contact mode, the same
        iteration runs as colour-batched Matrix kernels instead of a Python loop.
        batched defaults to the module flag for the serial A/B switch; the
        parallel path passes its noticeboard-carried value explicitly, since a
        worker sub-interpreter cannot see the main interpreter's module global.
        FRICTION constraints are swept apex-first (gravity-aligned, top-down) with
        accumulated PGS; the running impulses lambda reset each call (no
        warm-start). Every other mode keeps the stock per-iteration loop.
    """
    if batched is None:
        batched = use_batched_solver

    if batched and physics.mode.is_contact_mode:
        resolve_batched(physics, manifolds, num_velocity_iterations)
        return

    constraints = [physics.prepare_collision(*manifold) for manifold in manifolds]
    if physics.mode == PhysicsMode.FRICTION:
        constraints.sort(key=constraint_height)
        lambda_n = [[0.0] * len(c.contacts) for c in constraints]
        lambda_t = [[0.0] * len(c.contacts) for c in constraints]
        tangents = [build_tangent_data(c) for c in constraints]
        for _ in range(num_velocity_iterations):
            for k in range(len(constraints)):
                physics.apply_accumulated(constraints[k], lambda_n[k],
                                          lambda_t[k], tangents[k])
        return

    for _ in range(num_velocity_iterations):
        for constraint in constraints:
            physics.apply_collision(constraint)


def resolve_manifolds(physics: Physics,
                      pairs: List[Tuple[RigidBody, RigidBody]],
                      num_velocity_iterations: int, contacts: ContactSet = None,
                      batched=None) -> List[Manifold]:
    """Build, positionally correct, then velocity-solve one group of pairs.

    Description:
        The shared contact-resolution tail of every sub-step, serial or worker:
        build each pair's manifold at the current pose (dropping false positives),
        apply positional correction once, then iterate the velocity solver over
        the cached manifolds. Keeping it one function is what stops the serial
        driver and the two worker behaviors from drifting in how they sequence
        build -> separate -> resolve. batched is threaded to resolve_pair_list:
        None lets the serial path read the module flag; the workers pass their
        noticeboard-carried value explicitly.
    """
    manifolds = build_group_manifolds(pairs, contacts)
    for manifold in manifolds:
        separate_manifold(manifold)
    resolve_pair_list(physics, manifolds, num_velocity_iterations, batched)
    return manifolds


def solve_group_substep(physics: Physics, bodies: List[RigidBody],
                        pairs: List[Tuple[RigidBody, RigidBody]],
                        gravity, sub_dt: float, num_substeps: int,
                        num_velocity_iterations: int, contacts: ContactSet):
    """Advance one group of bodies over all sub-steps, separating integration from solve.

    Description:
        Each sub-step integrates the bodies once (per-body order preserved),
        builds every pair's contact manifold at the post-integration pose, then
        applies positional correction once before iterating the velocity solver
        over those cached manifolds. Building stays a pure geometry pass and
        separation is its own pass, so the manifold set is order-independent.
    """
    for _ in range(num_substeps):
        integrate_block(bodies, gravity, sub_dt)
        resolve_manifolds(physics, pairs, num_velocity_iterations, contacts)
