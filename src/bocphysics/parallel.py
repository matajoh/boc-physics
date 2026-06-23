"""Behavior-Oriented Concurrency scheduler for the parallel physics step.

Description:
    The serial engine solves every patch and seam on one thread. This module
    runs the same solver core across BOC workers: each patch's mutable state is
    one (N x 7) Matrix in a cown, and a behavior acquires that cown to integrate
    and resolve the patch in a worker sub-interpreter. Immutable geometry and the
    solve config ride the noticeboard, seeded once and cached per interpreter, so
    only the small state block crosses each frame. Per-cown FIFO ordering is the
    only synchronization: behaviors scheduled on a patch's cown run in schedule
    order, which is how successive sub-steps stay ordered without any barrier.
"""

from typing import List, NamedTuple, Optional, Tuple

from bocpy import (Cown, Matrix, notice_read, notice_seed, PinnedCown, start,
                   when)

from . import geometry, solver, transport
from .patches import build_partition, build_slab_partition
from .physics import Physics

GEOMETRY_KEY = "geometry"
GEOMETRY_VERSION_KEY = "geometry_version"
CONFIG_KEY = "config"

# the parallel default cut: K equal-population vertical slabs. Gate G3 measured
# this beating the loose-quadtree cut (~1.4-1.5x frame time, steal_failures
# halved) because vertical slabs follow gravity's stacking direction, so the
# seam colour count -- and thus the per-substep barrier depth -- drops ~9 -> ~2-4
DEFAULT_SLABS = 12

# floor on a slab's population: the slab count is capped at n_dynamic // this, so
# a small or sparse scene collapses to fewer, fuller slabs rather than one-body
# slabs that would turn every interior contact into a seam. At ~80 bodies this
# still yields the full DEFAULT_SLABS (80 // 6 = 13, capped to 12) -- the G3 point
MIN_SLAB_BODIES = 6

# each worker sub-interpreter imports this module fresh, so this cache is
# naturally per-interpreter: a body shell is rebuilt once per uid and reused
shell_cache = geometry.ShellCache()


class SolveConfig(NamedTuple):
    """Set-once solve parameters shared by every worker via the noticeboard.

    Description:
        This is a noticeboard payload, never a cown value, so carrying the
        gravity Matrix and the immutable Physics inside it is fine -- the
        runtime caches it per interpreter after the first read.
    """

    physics: Physics
    gravity: Matrix
    sub_dt: float
    num_velocity_iterations: int
    batched: bool


def shells_by_uid(geom, dyn_uids: List[int],
                  interior_uid_pairs: List[Tuple[int, int]]) -> Tuple[list, dict]:
    """Rehydrate this patch's dynamic and static shells, returning them by uid.

    Description:
        The dynamic shells come back in dyn_uids order so they line up with the
        state block rows. The statics are whatever uids the pairs reference that
        are not dynamic -- a static is only ever present because a contact needs
        it. Returns the dynamic shells and a uid -> shell map spanning both.
    """
    dyn_shells = shell_cache.shells(geom, dyn_uids)
    by_uid = {shell.uid: shell for shell in dyn_shells}
    dynamic = set(dyn_uids)
    static_uids = list({uid for pair in interior_uid_pairs for uid in pair}
                       - dynamic)
    for shell in shell_cache.shells(geom, static_uids):
        by_uid[shell.uid] = shell

    return dyn_shells, by_uid


def solve_intra_substep(state, pairs_block):
    """Integrate a patch's dynamics one sub-step and resolve its interior pairs.

    Description:
        The dynamic uids and their order come from the block's own uid column,
        so the block is the single source of truth for which bodies this patch
        owns. The interior pairs ride in their own cown as an (M x 2) uid block
        (or None when the patch has no interior pairs), reused across every
        sub-step. Reapplies the latest state first so any boundary edits made
        between sub-steps are picked up, integrates the dynamics, builds the
        interior manifolds, applies positional correction, then iterates the
        velocity solver over the interior and dynamic-static pairs at the post-
        integration pose, then writes the new state back for the next behavior.
    """
    geom = notice_read(GEOMETRY_KEY, {})
    shell_cache.evict_retired(geom, notice_read(GEOMETRY_VERSION_KEY, 0))
    config = notice_read(CONFIG_KEY)
    block = state.value
    interior_uid_pairs = transport.unpack_pairs(pairs_block)
    dyn_uids = transport.uids_of(block)
    dyn_shells, by_uid = shells_by_uid(geom, dyn_uids, interior_uid_pairs)
    transport.apply_state(dyn_shells, block)

    pairs = [(by_uid[ua], by_uid[ub]) for ua, ub in interior_uid_pairs]
    solver.integrate_block(dyn_shells, config.gravity, config.sub_dt)
    solver.resolve_manifolds(config.physics, pairs, config.num_velocity_iterations,
                             batched=config.batched)

    transport.store_state(dyn_shells, block)


def schedule_intra(state_cown, pairs_cown):
    """Schedule one intra-patch sub-step behavior on the patch's state cown."""
    @when(state_cown, pairs_cown)
    def _intra(state, pairs):
        solve_intra_substep(state, pairs.value)


def solve_boundary_substep(state_a, state_b, pairs_block):
    """Resolve the cross-patch pairs that stitch two patches at one sub-step.

    Description:
        Boundary pairs are dynamic-dynamic by construction, so no statics are
        replicated here. Both patches' latest state is reapplied to their shells,
        the owned seam pairs are built, positionally corrected, and resolved at
        the current pose, and both blocks are written back. There is no
        integration -- that already happened in the intra sub-step that ran first
        on each cown's FIFO. The pair block carries (uid_a, uid_b) in endpoint
        order so the contact normal is never flipped.

        Known divergence (Chunk-4 M1): the seam manifold is built here, after
        each patch has already velocity-solved its interior, so the restitution
        target sampled in restitution_bias sees an already-damped closing speed.
        The serial path builds every manifold up front at the post-integration,
        pre-resolve state. The two paths therefore agree at or below the
        restitution threshold (resting contacts, the stacking regime) but the
        decomposed seam suppresses restitution above it. The gap is quantified
        and locked by test_parallel.test_seam_decomposition_* across the
        threshold. Closing the gap is entangled with the seam pose (positional
        correction also runs before this build) and is deferred to the solver
        architecture work, not patched in isolation.
    """
    geom = notice_read(GEOMETRY_KEY, {})
    shell_cache.evict_retired(geom, notice_read(GEOMETRY_VERSION_KEY, 0))
    config = notice_read(CONFIG_KEY)
    block_a = state_a.value
    block_b = state_b.value
    boundary_uid_pairs = transport.unpack_pairs(pairs_block)

    shells_a = shell_cache.shells(geom, transport.uids_of(block_a))
    shells_b = shell_cache.shells(geom, transport.uids_of(block_b))
    transport.apply_state(shells_a, block_a)
    transport.apply_state(shells_b, block_b)

    by_uid = {shell.uid: shell for shell in shells_a}
    by_uid.update({shell.uid: shell for shell in shells_b})
    pairs = [(by_uid[ua], by_uid[ub]) for ua, ub in boundary_uid_pairs]
    # seam restitution is sampled in this build, post interior solve, so it is suppressed above threshold (Chunk-4 M1)
    solver.resolve_manifolds(config.physics, pairs, config.num_velocity_iterations,
                             batched=config.batched)

    transport.store_state(shells_a, block_a)
    transport.store_state(shells_b, block_b)


def schedule_boundary(state_a_cown, state_b_cown, pairs_cown):
    """Schedule one boundary sub-step behavior on the two patches' state cowns."""
    @when(state_a_cown, state_b_cown, pairs_cown)
    def _boundary(state_a, state_b, pairs):
        solve_boundary_substep(state_a, state_b, pairs.value)


def scatter_results(state_cowns, engine):
    """Scatter every patch's final block into the authoritative bodies by uid.

    Description:
        Runs on the main interpreter (the writeback holds a pinned cown), so it
        mutates the real RigidBody objects directly. Each block's uid column is
        the source of truth for which bodies it owns, so a block is matched to
        its bodies by uid, never by position in the body list. The out-of-bounds
        sweep runs here too, mirroring the tail of the serial step.
    """
    by_uid = {body.uid: body for body in engine.bodies if body.physics}
    for state in state_cowns:
        block = state.value
        bodies = [by_uid[uid] for uid in transport.uids_of(block)]
        transport.apply_state(bodies, block)

    engine.remove_outside()


def schedule_writeback(state_cowns, engine_pinned):
    """Schedule the single pinned writeback once every patch's solve is done.

    Description:
        The list of every patch state cown is one argument and the pinned engine
        cown is the second, so per-cown FIFO places this behavior dead last: it
        cannot run until every prior intra and boundary behavior on every patch
        cown has completed. The pinned cown forces it onto the main interpreter.
    """
    @when(state_cowns, engine_pinned)
    def _writeback(states, eng):
        scatter_results(states, eng.value)


def seam_groups(partition):
    """Group boundary pairs by the unordered patch pair they connect.

    Description:
        Many seams can connect the same two patches; resolving them in one
        behavior per patch pair keeps each pair of state cowns acquired together
        once per sub-step instead of once per seam. The key is the sorted patch
        index pair; the value is the list of (uid_a, uid_b) seams in endpoint
        order so the contact normal is never flipped.
    """
    groups = {}
    for boundary in partition.boundary_pairs:
        key = (min(boundary.patch_a, boundary.patch_b),
               max(boundary.patch_a, boundary.patch_b))
        a, b = boundary.pair
        groups.setdefault(key, []).append((a.uid, b.uid))

    return groups


def colored_seam_order(keys, num_patches):
    """Order seam keys so seams sharing a patch land in different colour batches.

    Description:
        Each seam locks two patch cowns, so the order seams are scheduled sets
        the seam layer's critical-path depth -- arbitrary order builds a chain
        that runs nearly one seam at a time (the dining-philosophers effect).
        Greedily edge-colours the seam graph (a colour per patch pair, no two
        seams sharing a patch share a colour), then emits all colour 0, then
        colour 1, and so on. A whole colour is an independent set, so it can
        occupy every patch-cown head at once, cutting the depth. The colouring
        is a pure function of the sorted keys, so the order is deterministic
        and worker-count independent. This reorders only when seams resolve,
        never which pairs resolve, so it is not a physics change on its own;
        it changes the parallel solve's resolution order like any reschedule.
    """
    used = [set() for _ in range(num_patches)]
    colors = {}
    for (i, j) in sorted(keys):
        c = 0
        while c in used[i] or c in used[j]:
            c += 1
        colors[(i, j)] = c
        used[i].add(c)
        used[j].add(c)
    return sorted(keys, key=lambda k: (colors[k], k))


class ParallelStepper:
    """Drives the physics step across BOC workers by composing a PhysicsEngine.

    Description:
        The stepper owns no physics of its own: it reuses the engine's broad
        phase, solver core, and bounds, and only changes WHERE the per-patch
        solve runs. Each frame it cuts the world into patches, packs each
        patch's mutable state into one cown, seeds the immutable geometry, then
        fans out the intra and boundary behaviors across every sub-step in
        dependency order. Per-cown FIFO sequences the sub-steps; a single pinned
        writeback scatters the final blocks back onto the authoritative bodies.
        A fixed timestep keeps sub_dt constant so the solve config is seeded once.
    """

    def __init__(self, engine, coarsen: float = 2.0, threshold: int = 8,
                 num_slabs: Optional[int] = DEFAULT_SLABS,
                 min_slab_bodies: int = MIN_SLAB_BODIES):
        """Compose a stepper over an engine, with the loose-cut tuning knobs.

        num_slabs selects the partition strategy: an int >= 1 cuts the world into
        that many equal-population vertical slabs (the default, DEFAULT_SLABS),
        which keeps the seam graph -- and thus the parallel barrier depth -- far
        shallower under gravity; None selects the loose-quadtree cut instead, the
        Phase 5 fallback the slab cut superseded at Gate G3. min_slab_bodies
        floors each slab's population so a small scene collapses to fewer, fuller
        slabs instead of degenerate one-body slabs (ignored for the quadtree cut).
        """
        if num_slabs is not None and num_slabs < 1:
            raise ValueError(f"num_slabs must be >= 1 or None, got {num_slabs}")

        self.engine = engine
        self.coarsen = coarsen
        self.threshold = threshold
        self.num_slabs = num_slabs
        self.min_slab_bodies = min_slab_bodies
        self.dt = 1 / 60
        self.engine_pinned = None
        self.geometry_version = None
        self.geometry_epoch = 0

    def begin(self, worker_count=None, dt: float = 1 / 60):
        """Start the runtime, pin the engine, and seed the set-once solve config."""
        self.dt = dt
        start(worker_count=worker_count)
        self.engine_pinned = PinnedCown(self.engine)
        sub_dt = dt / self.engine.num_substeps
        config = SolveConfig(self.engine.physics, self.engine.gravity, sub_dt,
                             self.engine.num_velocity_iterations,
                             solver.use_batched_solver)
        notice_seed(CONFIG_KEY, config)

    def seed_geometry(self):
        """Republish the immutable geometry only when the body set has changed.

        Description:
            Geometry is fat and cached per interpreter, so re-seeding every frame
            would bump the noticeboard version and force a re-pickle. The body set
            changes only when bodies are added or removed, which the (next_uid,
            count) pair uniquely flags, so geometry is republished only then.
        """
        version = (self.engine.next_uid, len(self.engine.bodies))
        if version != self.geometry_version:
            self.geometry_epoch += 1
            notice_seed(GEOMETRY_KEY, geometry.build_geometry(self.engine.bodies))
            notice_seed(GEOMETRY_VERSION_KEY, self.geometry_epoch)
            self.geometry_version = version

    def cut_partition(self):
        """Cut the world into patches by the selected strategy (slabs or quadtree)."""
        engine = self.engine
        if self.num_slabs is None:
            return build_partition(engine.bodies, engine.collisions,
                                   engine.detection.box, self.coarsen,
                                   self.threshold)

        return build_slab_partition(engine.bodies, engine.collisions,
                                    engine.detection.box, self.num_slabs,
                                    min_slab_bodies=self.min_slab_bodies)

    def step(self):
        """Run one parallel frame: cut, pack, seed, fan out, schedule writeback.

        Description:
            Mirrors the serial engine.step prologue (swept AABBs then broad
            phase) but partitions the candidate pairs into patches and runs each
            patch's solve on a worker. All M sub-steps are scheduled up front;
            per-cown FIFO orders them. Seams are emitted in colour order so the
            seam layer stays shallow (see colored_seam_order). The writeback is
            scheduled last and runs on main. Caller drives completion via quiesce
            (headless) or pump (GUI). Returns True when a writeback was scheduled,
            False when the frame was empty so the caller keeps in-flight accurate.
        """
        engine = self.engine
        engine.contacts.clear()
        engine.update_swept_aabbs(self.dt)
        engine.collisions.clear()
        engine.broad_phase()
        partition = self.cut_partition()
        if not partition.patches:
            return False

        self.seed_geometry()
        state_cowns = []
        intra_pairs = []
        for patch in partition.patches:
            state_cowns.append(Cown(transport.pack_state(patch.bodies)))
            uid_pairs = [(a.uid, b.uid) for a, b in patch.interior_pairs]
            intra_pairs.append(Cown(transport.pack_pairs(uid_pairs)))

        groups = seam_groups(partition)
        seam_order = colored_seam_order(list(groups), len(partition.patches))
        seams = [(key, Cown(transport.pack_pairs(groups[key])))
                 for key in seam_order]

        for _ in range(engine.num_substeps):
            for state, pairs in zip(state_cowns, intra_pairs):
                schedule_intra(state, pairs)
            for (i, j), pairs in seams:
                schedule_boundary(state_cowns[i], state_cowns[j], pairs)

        schedule_writeback(state_cowns, self.engine_pinned)
        return True
