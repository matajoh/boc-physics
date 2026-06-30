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

B3. Pool-source build_contacts' remaining scalar reads.
    Feed GeometryPool.sync_from() from the State block (dyn) + static table (B2).
    Replace the broad-phase a.aabb.disjoint with an aabb derived from pool columns
    (or precomputed per frame), the circle detect_collision pose reads, and the
    circle find_contact_points `c - a.position` with pool-column reads. After B3,
    build_contacts reads NO scalar body pose. GATE: golden bit-exact + settling.

B4. Row-index the constraints; solver writes the pool.
    ContactConstraint carries pool row indices (idx_a, idx_b) alongside the body
    refs. Serial apply_positional_impulse / apply_velocity_impulse and the colour
    kernel scatter impulses onto the PERSISTENT State block by row (kernel already
    computes idx_a/idx_b + scatter-adds; point it at the persistent block instead
    of pack_poses/pack_bodies-then-unpack). Mirror still writes bodies. GATE:
    serial golden bit-exact, colour settling-band.

B5. Integrate + derive on the State block in place.
    integrate_block mutates State columns directly; snapshot_poses reads State
    columns; derive_velocities reads/writes State columns. Bodies still mirrored.
    GATE: golden bit-exact.

B6. Persist State across substeps; drop the per-substep mirror (THE WIN).
    solve_group_substep keeps the State block authoritative for all N substeps;
    scatter to bodies ONCE at frame end (for render/broad phase). In parallel.py
    solve_intra_substep, apply_state/store_state move OUT of the per-substep body
    and run ONCE per behavior entry/exit. GATE: golden bit-exact + settling; and
    the transport METRIC — pack/store/apply us and patch send/recv us must drop
    ~Nx (8 substeps -> 1). Report ms/frame for serial, batched, and BOC paths.

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
