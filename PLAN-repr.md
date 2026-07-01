# PLAN — SoA state pool + flat-row geometry (representation rework)

Goal: cut the per-substep Matrix marshalling/transport tax. Bodies stop holding
scalar attributes; mutable state lives in one columnar pool that integrates in
place and crosses BOC workers by pointer. Geometry follows the same SoA shape.

Sequencing resolves the speed/usability/conservative fork. The solve
(solve_positions/apply_positional_impulse, derive_velocities, build_contacts,
solve_velocities) reads AND mutates body.position/angle/velocity every substep, so
the pool cannot be authoritative until those reads/writes move onto the block.
Geometry pool went FIRST (Phase 3, done) so the narrow-phase distance reads source
the pool; the structural win is the B-bridge below, which persists ONE block across
substeps behind a write-through mirror — it does NOT require read-view aliasing.

Invariants EVERY step: 827 goldens bit-exact, flake8 rc=0, parity fuzz clean,
single-line comments <=120ch. Bench gate = profile_narrow.py (seeded; promote to
bench/) seeds 7/8/9 over N>=3 runs, noise band; drop_box UNSEEDED=trend. ONE
canonical body->row map shared by state+geometry pools; statics get rows but are
never integrated. Each step independently green/revertible. Agent never commits.

## Transport cost (track explicitly — a HUGE win if it lands)
Two halves: (1) WIRE = the (N,7) Matrix crosses by pointer ~0.2us flat (XIData);
preserved. (2) MARSHALLING = pack/store/apply box bodies field-by-field O(N) EVERY
substep — this is the cost the bridge kills. B1-B5 move each read/write onto the
State block behind a write-through body mirror; B6 removes the mirror so
apply_state/store_state run ONCE per behavior instead of once per substep (8x ->
1x) and integrate/solve mutate pool slices in place. (Geometry-by-pointer on the
noticeboard is a later, separate win; not on the B-bridge critical path.) METRIC:
report pack/store/apply us, noticeboard pickle us, patch send/recv us per step
alongside ms/frame for serial, batched, AND BOC — not just FPS.

## Phase 0 — Baseline (no code)
1. Record pytest (827), flake8 src test bench, drop_box seeds 7/8/9, profile_narrow
   ms/frame + build_contacts% + transport breakdown. Note pre-existing failures.

## Phase 1 — SKIPPED (seam already factored)
2. transport pack_state/store_state/apply_state already ARE the gather/scatter
   seam; integrate_block already gathers->3 ops->scatters. No measurable win; the
   setter rewrite (~6 in-place sites, bit-exact) was done instead as Phase 2 prep.

## Why the geometry pool must be ON State (rationale, still load-bearing)
The structural win needs ONE authoritative columnar pool that both contact
batchers read by row, NOT a throwaway GeometryPool rebuilt from scalar bodies each
substep. Poly-poly CANNOT be skipped: its scan_polygon_edges/edge_point_distances
reads are exactly what would otherwise pin bodies — Phase 3 already routed those
through the pool, which is why the bridge below is now mostly wiring, not new math.
The remaining pin is that the pool's POSE still comes from scalar bodies and the
solve still mutates scalar bodies; B1-B6 move both onto the State block. Statics
transform once at rebuild; only dynamics rewrite rows per frame.

## Phase 3 (DONE) — Narrow phase vectorised via a standalone GeometryPool
9. DONE: transport.GeometryPool (polys+statics, circles excluded; vmax/nmax pad,
   replicate vertex0/zero normals; row_of; sync rewrites rows). 889 pass, flake8 ok.
10. DONE: circle-poly reads geom_x/geom_y/norm_x/norm_y .take(rows,0); vertex0/zero
    pad bit-exact; scalar broadcast (_BIG, sign). 2.19s->0.65s cumtime, 113ms/frame.
10b. DONE: poly-poly batched via pool (both bodies' rows); intersect_polygon_polygon
    1.77s->0, build_contacts 11.3s->9.2s, 106ms/frame.
10c. DONE: contact-point gen (batched_contact_points) vectorised + packed as one
    (N,13) Matrix in stride-6 [count | px,py,ra,rb per point] layout; build_contacts
    decodes by row; contact_velocity caches the static zero. 900 pass, golden bit-
    exact. Matrix.vector gone from the hot path; e2e drop_box flat (alloc was a
    single-digit % of frame).
11. Cleanup + editor-lens; refresh docstrings + README bench stats; flake8 rc=0.

## Where we are now (2026-06-30) — the divergence to close
Phase 3 vectorised the narrow-phase MATH, but it did NOT advance pool-authority.
GeometryPool is a STANDALONE class, REBUILT from scalar bodies every build_contacts:
sync() reads p.position.x/.y and p.angle per poly, and constraints bind to body
OBJECTS. The serial AND colour kernels still read/mutate scalar body fields each
substep (xpbd_kernel packs scalar->SoA, solves, scatters SoA->scalar TWICE per
substep). So the BOC per-substep apply_state/store_state scatter+gather (the real
transport MARSHALLING cost, NOT the flat ~0.2us wire) is unchanged. The pool the
plan wanted "ON State" got built as a derived, throwaway-per-substep copy instead.

The blocker is therefore STILL the one Phase 0 named: build_contacts reads body
pose scalars mid-substep. Concretely, every scalar body touch left in a substep:
  - snapshot_poses: body.position.x/.y, body.angle
  - integrate_block: reads + writes position/velocity/angle/spin
  - build_contacts: a.aabb.disjoint(b.aabb); GeometryPool.sync() pose reads;
    circle detect_collision; circle find_contact_points `c - a.position`;
    ContactConstraint(a, b, ...) binds body objects
  - solve_positions/apply_positional_impulse: a.move / a.rotate_to
  - derive_velocities: position/angle read, velocity/spin write
  - solve_velocities/apply_velocity_impulse: velocity/spin write
The poly-poly distance reads already source the pool; the rest do not.

## Bridge to Phase 2 — concrete, incremental, each step independently green
Strategy: keep scalar bodies as a WRITE-THROUGH MIRROR while we move each read/
write onto the State block, so every step stays golden bit-exact (serial) /
settling-band (colour). The mirror is removed only at the final step, which is
when the per-substep marshalling actually collapses. State scaffold already exists
(transport.State: authoritative (N,7), uid->row, gather/scatter) but is unused by
the live path — these steps put it on the critical path.

B1. Pose-source the geometry pool (refactor, bit-exact). DONE.
    Split GeometryPool.sync() into "read pose -> (px,py,cos,sin) arrays" and "apply
    arrays -> rotate/translate rows". Add a sync_from(px,py,angle) that takes pose
    columns. Keep the body-reading path feeding identical values for now. GATE:
    golden bit-exact, flake8. WIN: geometry refresh no longer HARD-WIRED to bodies.
    LANDED: sync()=body-sourced, sync_from(px,py,angle)=column-sourced, both fill
    self.px/py/cos/sin then _apply_pose() (cos/sin per-element math for bit-exact).
    New fuzz test_geometry_pool_sync_from_matches_body_sync (30 seeds). 930 pass.

B2. One canonical row space (geometry ON State, statics get rows). DONE (revised).
    Goal: the pool can supply EVERY poly's current pose by row without a scalar
    read. REVISED DESIGN (cleaner than literal statics-in-block): statics are NOT
    appended to the integrated (N,7) block (that would create dead rows integrate
    must skip). Instead the dynamics block stays dynamics-only; GeometryPool keeps
    each static poly's pose in its own px/py/cos/sin (seeded at rebuild) and bridges
    to the dynamics block BY UID. New sync_from_block(block, row_of): dynamic polys
    read pose from block[row_of[uid]] (POSITION/ANGLE cols), static polys keep
    rebuild pose; cos/sin per-element math for bit-exact. GATE: round-trip fuzz
    test_geometry_pool_sync_from_block_matches_bodies (30 seeds, mixed dyn/static)
    == body-sourced reference; golden bit-exact; flake8. 960 pass. NOTE the two row
    spaces (State.row_of dyn, GeometryPool.row_of polys) are bridged by uid, not
    merged — accepted (uid dict lookup is cheap vs the matrix ops).

B3. Pool-source build_contacts' remaining scalar reads. (EXPANDED — 5 substeps.)
    Goal: after B3, build_contacts reads NO scalar body pose. This is the first
    STRUCTURALLY INVASIVE step (B1/B2 were additive scaffolding). build_contacts is
    the SHARED narrow phase: serial xpbd.solve_substep, colour xpbd_kernel.solve_substep
    (line ~220), AND BOC parallel.solve_intra_substep (line ~164) all call it, so EVERY
    B3 substep gates serial=golden bit-exact AND colour/BOC=settling-band.

    Scalar body-pose reads left in build_contacts today (the B3 hit list):
      (1) broad phase: `a.aabb.disjoint(b.aabb)` — aabb reads body pose (circle
          position.x/.y +/- radius; polygon over transformed_vertices).
      (2) GeometryPool(polys) ctor still body-sources pose via sync().
      (3) circle SAT: batched_circle_circle / batched_circle_polygon read
          a.position.x/.y (collisions.py); intersect_* read circle.position.
      (4) circle find_contact_points: `a.position + normal*radius`, `c - a.position`.
      (5) batched_contact_points: pos_a[i]=a.position, pos_b[i]=b.position lever arms
          (contacts.py ~104/106); and the serial circle branch `c - a.position`,
          `c - b.position` lever arms in build_contacts itself.

    WHERE THE BLOCK ENTERS (the key wiring decision): the serial path has NO State
    block in scope. Per the write-through MIRROR strategy, pack a fresh State block
    from bodies AFTER integrate_block (build_contacts runs at the NEW pose), before
    build_contacts, and thread it in as a new param. In B3 the block still MIRRORS
    bodies (body-authoritative); B5 moves integrate onto the block so the per-substep
    pack disappears. A POSE SOURCE covers statics: dynamics (incl. dynamic circles)
    resolve via state.row_of -> block POSITION cols; statics (absent from the dyn
    block) resolve via a per-frame static pose table keyed by uid (poly poses already
    on the GeometryPool from B2; add a static-CIRCLE pose table). Helper
    `pose_of(uid)` -> (x, y) hides the dyn/static split so call sites read one way.

    B3a. Thread a State block through the substep into build_contacts (no behavior
         change). DONE. build_contacts gained `state=None`; when given it calls
         transport.assert_block_mirrors(block, row_of, eligible bodies) (exact
         float compare, statics skipped) as the bridge safety net. Helper +
         fuzz tests (30 seeds pass, 1 divergence raises) added. 990 pass, golden
         bit-exact, flake8 clean. FINDING (defers part of the plan): State is
         uid-keyed but solve_substep is called in micro-tests with uid-less bodies
         (production bodies always carry a scene uid). So LIVE State construction
         does NOT belong inside solve_substep — it must be owned where uids exist.
         The state param is in place but unfed by the live path until B3b builds
         and threads the State from the right level (engine/solve_group_substep).
    B3b. Pool pose from the block. DONE. engine.solve_substep (serial branch only)
         builds `state = transport.State(bodies)` where uids are guaranteed
         (add_body assigns them) and threads it through solve_group_substep ->
         solve_substep -> build_contacts. solve_substep calls state.gather() after
         integrate_block (write-through refresh of the block from bodies), then
         build_contacts calls geom.sync_from_block(state.block, state.row_of) so
         dynamic poly pose comes from the block; statics keep ctor pose. Micro-test
         callers of solve_group_substep pass no state (None) -> body-sourced, so
         the uid-less unit tests are untouched. THREE bugs the live wiring exposed,
         all root-caused: (1) _apply_pose crashed on an empty pool (all-circle
         eligible set) -> no-op when rows == 0; (2) State.gather/scatter crashed on
         an all-static group (block is None) -> no-op when block is None; (3) the
         `polys` dedup `{p.uid: p}` COLLAPSES bodies that share uid=None -- a latent
         build_contacts fragility that only bites uid-less bodies. test_solver drove
         engine.solve_substep with uid-less bodies; giving BOTH the ref (engine,
         block-sourced) and cand (core, body-sourced) groups real uids turned those
         two tests into a genuine block-vs-body cross-check that now passes bit-exact.
         990 pass, golden bit-exact, flake8 clean.
    B3c. DONE. Broad-phase cull off the block, not a.aabb. Chose a conservative
         bounding-circle box over the planned tight pool-aabb table: `_broad_box`
         returns AABB(centre +/- body.radius) with the dynamic centre read from
         state.block[row] (keyed by uid via state.row_of) and the static centre
         read from the immutable body.position (statics never integrate). The box
         is rotation-invariant so it needs no angle. Replaced
         `a.aabb.disjoint(b.aabb)` with id()-keyed `_broad_box` boxes over the
         physics-bearing candidate set (id() dodges the uid=None landmine for
         state=None micro-tests). Why bit-exact despite a LOOSER box: the cull is
         perf-only -- it changes neither the constraint SET nor ORDER, because (a)
         a colliding pair (depth>0) always passes any conservative cull (the
         bounding circle provably contains the polygon, so overlapping bodies
         always have overlapping boxes), (b) extra non-colliding pairs that slip
         through return None from SAT and are skipped, (c) `eligible` preserves
         `pairs` order so colliding-pair relative order is invariant. Verified
         nothing downstream needs a.aabb's update_transform side effect: the geom
         pool builds its own world vertices from the block, find_contact_points
         reads pose+geom, and the detect_collision fallback is dead code. Added 60
         conservativeness fuzz tests (test_broad_box_never_rejects_a_real_overlap):
         random near-origin circle/rect pairs assert `not box_a.disjoint(box_b)`
         whenever detect_collision finds depth>0. 1050 pass / 1 skip, golden
         bit-exact, flake8 clean, drop_box bench ~2.5 ms/frame (no regression).
    B3d. DONE. Circle SAT pose from the block. New collisions._circle_center(circle,
         state) returns the centre (x, y): dynamic from state.block[row] by uid,
         static from circle.position (statics never integrate). batched_circle_circle
         and batched_circle_polygon gained a trailing `state=None` and source every
         circle centre through it; radius stays a body constant. xpbd threads state
         through _batch_circle_collisions into both. Bit-exact: the block mirrors the
         body so block[row] == position.x/.y exactly. GATE: 1052 pass / 1 skip, golden
         bit-exact, flake8 clean. Two parity tests (test_collisions): 2000 random
         circle-circle and circle-poly pairs, body-sourced == block-sourced bit-for-bit.
    B3e. DONE. All remaining build_contacts pose reads off the block. New
         transport.block_center(body, state) returns a Matrix centre (the slice
         read state.block[row, POSITION] -- one C row copy, no per-float box) for
         lever arms; dynamic from the block by uid, static from body.position.
         contacts.find_contact_points (circle position +/- normal*radius) and
         contacts.batched_contact_points (poly lever-arm centres pos_a/pos_b) gained
         state and source centres through block_center. build_contacts' circle branch
         computes ca/cb via block_center. The completeness grep ALSO surfaced
         polygon-orientation centres in the batched SAT (batched_circle_polygon poly
         centre, batched_polygon_polygon a/b centres) -- block-sourced too via a
         renamed collisions._body_center (was _circle_center; now generic, keys on
         row_of membership NOT body.physics so it is robust to bodies lacking the
         attribute). batched_polygon_polygon gained state; xpbd threads it through.
         After B3e the LIVE build_contacts path touches NO dynamic body pose; the
         only residual collisions.py .position reads are the non-batched oracle
         (detect_collision / intersect_*, used by tests + spawn; the build_contacts
         fallback to detect_collision is dead -- resolved covers every eligible idx)
         and the static fallback (valid forever). contacts.py has ZERO body-pose
         reads. GATE: 1052 pass / 1 skip, golden bit-exact, flake8 clean.

    OPEN for B3: confirm dynamic-circle rows resolve in state.row_of (circles carry
    pose in block cols 1,2 but have NO geometry row — they must still get a State
    row). Static-circle pose ownership RESOLVED in B3c: read the immutable static
    body.position directly (valid forever -- statics never integrate), no table.

B4. Row-index the constraints; the serial solver writes the State block.
    SCOPE NARROWED: the BOC parallel path (parallel.solve_intra_substep) calls the
    SERIAL xpbd.solve_substep per patch, NOT the colour kernel. So the B6 win only
    needs the SERIAL path on the block. The colour kernel (xpbd_kernel, engine
    batched=True path) keeps its own per-substep pack_poses/pack_bodies and is OFF
    the critical path -- left as-is in B4 (optional later migration). Substeps:

    B4a. ContactConstraint gains idx_a, idx_b (Optional[int] row indices into the
         State block, or None for a static / when state is None). build_contacts
         populates them via state.row_of.get(uid). Update the 3 positional unpack
         sites (xpbd.solve_positions, xpbd.solve_velocities, xpbd_kernel.pack_colour
         -- the kernel just ignores the new fields, it builds its own id()-based
         rows). Pure plumbing, fields unused -> golden bit-exact, colour unaffected.
         DONE: 1052 pass / 1 skip, flake8 clean; added test_build_contacts_row_
         indices_match_state_rows (idx == row_of with state, None without).
    B4b. Serial apply_positional_impulse ALSO scatters the SAME pose delta onto
         state.block[idx] (mirror) when the block + idx are passed; bodies are still
         written unchanged so the body arithmetic (and thus the golden) is byte-
         identical. solve_positions threads state.block + c.idx_a/c.idx_b. After the
         position pass, assert_block_mirrors(block, row_of, bodies) confirms the
         scattered block POSITION/ANGLE equals the bodies (the existing build_contacts
         assert runs post-gather so it cannot catch a bad scatter -- this one runs
         BEFORE the next gather, so it is the real proof). The block writes are
         REDUNDANT in B4 (next substep's state.gather() re-mirrors after integrate) --
         they exist to establish + prove the row-scatter machinery so B5 (integrate on
         block, drop gather) and B6 (drop body writes) can build on a verified block
         write. GATE: golden bit-exact.
         DONE: 1053 pass / 1 skip, golden bit-exact, flake8 clean; the in-substep
         assert fires on every serial engine frame; added test_position_pass_mirrors_
         poses_onto_the_state_block (120-frame pile, body poses identical with/without
         state, block mirrors bodies at the end).
    Delta arithmetic (bit-exact mirror): position body a.move(P), P=impulse*-inv_m
    -> block[idx,POSITION.start]+=P.x, [+1]+=P.y (block row started == body.position
    via gather, same float added -> equal). Angle body a.rotate_to(a.angle - X),
    X=r_a.cross(imp)*inv_I -> block[idx,ANGLE]-=X (block ANGLE started == a.angle).
    Hoist P and X to locals, apply the identical value to both body and block.
    VELOCITY is NOT mirrored in B4b: the block VELOCITY/SPIN cols are set at gather
    time (post-integrate) and derive_velocities overwrites the BODY velocity without
    touching the block, so the block velocity is stale at solve_velocities entry.
    The velocity scatter (apply_velocity_impulse -> VELOCITY cols 3,4 / SPIN col 6)
    therefore lands in B5, paired with derive_velocities becoming block-aware.

B5. Integrate writes the State block; drop the gather (block becomes load-bearing).
    SCOPE NOTE: B5 keeps the bodies the authoritative READ surface (still mirrored)
    and migrates only the WRITE side onto the block, plus integrate reading the
    block. That is all the serial golden needs: bodies stay in lockstep via the
    mirror, so contact_velocity / derive / snapshot may keep reading bodies. The
    block-READ migration (contact_velocity / derive / snapshot sourced from the
    block) is only load-bearing once the body mirror is dropped, so it lands in B6
    with the parallel rewire. Substeps:

    B5a. Velocity-write mirror. derive_velocities and apply_velocity_impulse ALSO
         write block VELOCITY (cols 3,4) / SPIN (col 6) when block + idx given (the
         deferred half of B4b). solve_velocities threads state.block + c.idx_a/idx_b;
         derive_velocities threads state (uid -> row). With gather() still present
         these writes are REDUNDANT (next gather re-mirrors) -- a velocity mirror
         assert after solve_velocities is the proof they match the bodies bit-for-
         bit. GATE: golden bit-exact.
         DONE: 1054 pass / 1 skip, golden bit-exact, flake8 clean. assert_block_mirrors
         extended to VELOCITY/SPIN (passes at all sites: post-gather, post-position,
         post-velocity); derive + apply_velocity_impulse assign the SAME nv/av to body
         and block (no recompute -> no FMA-rounding risk). test_position_pass_mirrors
         now also asserts velocity bit-identical with/without state.
    B5b. Integrate on the block. solver.integrate_block_state(block, gravity, dt)
         reads block VEL/POS/ANGLE/SPIN, runs the same 3 batched ops, writes block
         VEL/POS/ANGLE, and mirrors the rows onto the bodies (bit-identical to
         integrate_block). solve_substep calls it instead of integrate_block when
         state is given. gather() is KEPT here (now redundant -- integrate already
         mirrored bodies == block). GATE: golden bit-exact.
         DONE: 1054 pass / 1 skip, golden bit-exact, flake8 clean. integrate_block_state
         reads block via 1-wide slices (ANGLE/SPIN); guarded on state.block is not None
         so the all-static (block None) scene falls back to integrate_block(bodies).
         The gravity step stays velocity += dt * gravity (gravity is a (1,2) broadcast
         row, not a same-shape scaled_add x; two-round, bit-exact with .add(gravity*dt)).
    B5c. Drop the top-of-substep state.gather(). The block is now maintained purely
         by integrate (B5b) + the position scatter (B4b) + the velocity scatter
         (B5a) -- no body->block refresh. Bodies stay == block via the per-write
         mirror, asserted every substep (position after solve_positions, velocity
         after solve_velocities). Golden staying bit-exact with NO gather is the real
         cross-substep proof that the block is a faithful authoritative store.
         GATE: golden bit-exact.
         DONE: 1054 pass / 1 skip, golden bit-exact, flake8 clean. Cross-substep proof
         holds: block carries VEL/POS/ANGLE/SPIN across substeps purely via mirror
         writes. State.gather() is now ORPHANED (no callers) but KEPT -- symmetric with
         scatter() (used by test_transport) and a valid State op; remove decision deferred.

B6. Persist State across substeps; drop the per-substep mirror (THE WIN).
    Migrate every pose/velocity READ and every block WRITE onto the block so the
    solver needs only the bodies' CONSTANTS (inv_mass, inv_inertia, radius,
    geometry). Each read/write migration stays golden bit-exact in SERIAL because
    the body mirror still holds (block == body), so B6a-B6c are gated purely by the
    serial golden and touch no parallel code. Only the final step (B6d) flips the
    parallel path and is gated by the settling band + metrics. KEY INVARIANT: the
    block writes must be computed from BLOCK reads (not body reads) so the block is
    authoritative independent of the (soon-stale) shells; in serial block == body so
    this stays bit-exact. Substeps:

    B6a. contact_velocity reads block VELOCITY/SPIN. Add block/idx params (default
         None); when given, lv = block[idx, VELOCITY] (1,2 slice), av = block[idx,
         SPIN] scalar, return lv + av * r.perpendicular(); static still _ZERO_VELOCITY.
         Thread block+idx through relative_normal_velocity (build_contacts passes
         state.block + idx_a/idx_b, computed once per hit) and solve_velocities'
         inner contact_velocity pair (already has block/idx_a/idx_b). GATE: serial
         golden bit-exact (block == body so the read is identical).
         DONE: 1054 pass / 1 skip, serial golden bit-exact, flake8 clean. contact_velocity
         reads block[idx, VELOCITY] (1,2 slice) + block[idx, SPIN] (scalar) when block+idx
         given; relative_normal_velocity + solve_velocities inner pair thread them.
    B6b. snapshot_poses + derive_velocities read block POSE. snapshot_poses(bodies,
         state=None): when state given, read (x, y, angle) from block[row, POSITION]/
         [row, ANGLE] per body; else the scalar body read. derive_velocities reads
         the CURRENT pose from block[row, POSITION]/[row, ANGLE] (not body.position)
         when state given. GATE: serial golden bit-exact.
         DONE: 1054 pass / 1 skip, serial golden bit-exact, flake8 clean. Both read scalar
         cells block[row, POSITION.start/+1] / [row, ANGLE]; None-row bodies fall back to the
         scalar body read; solve_substep passes state into the snapshot call.
    B6c. Write side computes from block reads. apply_positional_impulse and
         apply_velocity_impulse derive their block deltas/results from the block's
         CURRENT row values (block[idx, POSITION]/[ANGLE]/[VELOCITY]/[SPIN]), not the
         body's, so the block no longer depends on shell pose. Bodies are still
         updated (serial authoritative). FMA-sensitive: the block velocity write must
         reuse the SAME scaled_add form on the block's own velocity; in serial block
         == body so bit-exact. GATE: serial golden bit-exact (verify velocity ULP).
         DONE: 1054 pass / 1 skip, serial golden bit-exact, flake8 clean. Only
         apply_velocity_impulse changed: block[idx, vel] = block[idx, vel].scaled_add(...),
         block[idx, spin] -/+ dw (dw hoisted, reused for body). apply_positional_impulse
         UNCHANGED -- its block writes were already relative deltas (+= da, -= spin_a from
         impulse + constants, pose-independent). scaled_add is two-round (not FMA) and the
         body uses the same form, so block == body stays bit-exact.
    B6d. Parallel rewire (THE WIN). Add a lightweight State wrapper over an EXISTING
         block (no repack) -- e.g. transport.State.over(block, shells, uids) setting
         block/bodies/row_of directly. In solve_intra_substep: build the wrapper from
         state.value + dyn_shells + dyn_uids, REMOVE apply_state and store_state, and
         pass the wrapper into xpbd.solve_substep. The solver now reads/writes the
         block in place; the vestigial shell writes are never read (all reads are
         block-sourced after B6a-B6c) and never stored. solve_boundary_substep is
         UNCHANGED -- it reseeds shells from the (authoritative) block via its own
         apply_state and stores back, so it stays correct without the intra marshal.
         scatter_results still apply_state(engine_bodies, block) for render/broad
         phase. GATE: settling band (test_parallel) + determinism + serial golden
         still bit-exact. Report the transport METRIC: per-substep apply_state/
         store_state us -> 0 in intra (2*num_substeps marshal ops/patch/frame removed);
         ms/frame for serial, batched, and BOC paths.
    B6e (optional, profile-gated). Drop the vestigial shell writes in the block path
         (guard body writes on state is None) and migrate solve_boundary_substep the
         same way if the seam marshal shows up in the profile.

## Phase 2.5 — Worker shells (profile-gated, after B6)
8. Workers hold patch-local block, not global pool; per-patch rows. If shells
   dominate: alias patch row; else synced scalars. Measure churn (remove_outside).

## Risks
- Op reorder drift -> identical C op sequence + goldens/fuzz per step.
- Write-through mirror divergence -> B1-B5 write BOTH pool and bodies; a per-step
  assert (pool row == body field) catches any column-layout/order mismatch before
  B6 removes the mirror. The mirror is the safety net that keeps every step green.
- Pool churn (remove_outside/frame) -> measure; free-list/compaction.
- Copy views drop impulses -> bocpy slice reads COPY, slice-assign writes through;
  the pool MUTATES columns by slice-assign (no aliased views), so this is moot for
  the B-bridge. Re-audit only if a step reaches for a read-view alias.
- Row spaces: B2 collapses State (dyn) + GeometryPool (poly incl. static) into ONE
  uid->row map (dyn rows first, statics appended, statics never integrate). Circles
  have no geometry row -> they need their own pose source in the State block (cols
  1,2,5 already carry it); confirm circle rows resolve before B3.
- Worker rows != global row_of -> per-patch map (Phase 2.5).
- Colour kernel is settling-band, NOT golden bit-exact: B3/B4/B6 gate the colour
  path on the settling band, the serial path on the golden master. Keep them split.

## Open (escalate)
- AUDITED: bocpy slices COPY on read; slice-assign writes through. The pool path
  uses slice-assign mutation only (setter-form, ~6 in-place sites already rewritten
  as prep). Read-view aliasing (store=no-op) stays deferred to the upstream tensors
  roadmap; the B-bridge does NOT depend on it — it collapses marshalling by
  persisting ONE block across substeps, not by aliasing body fields onto pool rows.
