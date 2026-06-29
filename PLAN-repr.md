# PLAN — SoA state pool + flat-row geometry (representation rework)

Goal: cut the per-substep Matrix marshalling/transport tax. Bodies stop holding
scalar attributes; mutable state lives in one columnar pool that integrates in
place and crosses BOC workers by pointer. Geometry follows the same SoA shape.

Sequencing resolves the speed/usability/conservative fork. The solve
(solve_positions/apply_positional_impulse, derive_velocities, build_contacts,
solve_velocities) reads AND mutates body.position/angle/velocity every substep,
so the pool cannot be authoritative WITHOUT views. Phase 1 is a pure seam
refactor; the structural win is gated on views (Phase 2). Geometry pool last.

Invariants EVERY step: 827 goldens bit-exact, flake8 rc=0, parity fuzz clean,
single-line comments <=120ch. Bench gate = profile_narrow.py (seeded; promote to
bench/) seeds 7/8/9 over N>=3 runs, noise band; drop_box UNSEEDED=trend. ONE
canonical body->row map shared by state+geometry pools; statics get rows but are
never integrated. Each step independently green/revertible. Agent never commits.

## Transport cost (track explicitly — a HUGE win if it lands)
Two halves: (1) WIRE = the (N,7) Matrix crosses by pointer ~0.2us flat (XIData);
preserved. (2) MARSHALLING = pack/store/apply box bodies field-by-field O(N) every
substep. The plan kills #2: P1 collapses pack+store+apply to one seam; P2 views
make store=no-op, apply=dirty-bit, solve mutates pool slices in place; P3
geom_x/geom_y cross by pointer instead of pickling on the noticeboard. METRIC:
report pack/store/apply us, noticeboard pickle us, patch send/recv us per phase
alongside ms/frame — not just FPS.

## Phase 0 — Baseline (no code)
1. Record pytest (827), flake8 src test bench, drop_box seeds 7/8/9, profile_narrow
   ms/frame + build_contacts% + transport breakdown. Note pre-existing failures.

## Phase 1 — SKIPPED (seam already factored)
2. transport pack_state/store_state/apply_state already ARE the gather/scatter
   seam; integrate_block already gathers->3 ops->scatters. No measurable win; the
   setter rewrite (~6 in-place sites, bit-exact) was done instead as Phase 2 prep.

## Phases 2+3 ARE ONE MOVE — geometry into State, both batchers read pool
Geometry pool is NOT a standalone GeometryPool; it is geom_x/geom_y/norm_x/norm_y
rows ON State, alongside the (N,7) dynamics. The 7% circle-poly assembly win is
incidental. The POINT: build_contacts still reads body.transformed_vertices_block_
(per-body), which pins bodies authoritative and forces scatter mid-frame. Move
transformed geometry into State rows -> both circle-poly AND poly-poly contact
gen read take(idx) from the pool -> per-body position/geom reads vanish -> pool
stays authoritative -> gather once/scatter once. Poly-poly CANNOT be skipped: its
scan_polygon_edges/edge_point_distances reads are exactly what pin bodies.
Profile (seed7): circle-poly self 1.3s, poly-poly contact gen ~4.5s. Statics
transform once at rebuild; only dynamics rewrite rows per frame.

## Phase 3 (NOW) — Geometry pool ON State
9. DONE: transport.GeometryPool (polys+statics, circles excluded; vmax/nmax pad,
   replicate vertex0/zero normals; row_of; sync rewrites rows). 889 pass, flake8 ok.
10. DONE: circle-poly reads geom_x/geom_y/norm_x/norm_y .take(rows,0); vertex0/zero
    pad bit-exact; scalar broadcast (_BIG, sign). 2.19s->0.65s cumtime, 113ms/frame.
10b. DONE: poly-poly batched via pool (both bodies' rows); intersect_polygon_polygon
    1.77s->0, build_contacts 11.3s->9.2s, 106ms/frame. Next wall: contacts.py gen ~4s.
11. Cleanup + editor-lens; refresh docstrings + README bench stats; flake8 rc=0.

## Phase 2 (AFTER 3) — pool-authoritative substep loop (cut 9 boundaries to 1/frame)
Kernels already SoA. With geom in State, NO per-body reads remain; persist pool
across substeps, gather once/frame, scatter once. Bodies stale mid-frame OK.
Bit-exact gate, build on State scaffold.

## Phase 2.5 — Worker shells (profile-gated)
8. Workers hold patch-local block, not global pool; per-patch rows. If shells
   dominate: alias patch row; else synced scalars. Measure churn (remove_outside).

## Risks
- Op reorder drift -> identical C op sequence + goldens/fuzz per step.
- Pool churn (remove_outside/frame) -> measure; free-list/compaction.
- Copy views drop impulses -> write-through audit gates Phase 2.
- Two pools, two row spaces -> one body->row map; statics rows but no integrate.
- Worker rows != global row_of -> per-patch map.

## Open (escalate)
- AUDITED: bocpy slices COPY on read; slice-assign writes through. Views use
  setter-form (~6 in-place sites rewritten); write-through slices deferred to
  upstream tensors roadmap. Phase 1 ships regardless.
