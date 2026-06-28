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

## Phase 1 — seam refactor (modest serial saving)
2. transport.py: gather_into/scatter_from lifting pack/store/apply, same op order.
3. solver.py: integrate_block mutates pose/vel slices in place; saving = one pose
   round trip/substep (velocity gather is dead work — derive_velocities overwrites).
4. Gate: bit-exact, flake8, parity fuzz, profile seeds 7/8/9; (0,7) sentinel.

## Phase 2 — bodies-as-views via named State (the real win)
5. Write-through audit (blocker): move/rotate_to/scaled_add(in_place=True) need
   pool-aliasing views or setter rewrite; fuzz test in-place vel observed in pool.
6. Introduce State owning (N,7) pool, slices, canonical row_of.
7. View properties; setters write pool only (shadow-assert ladder if rewriting).
   ~9 call sites unchanged; position.x=5 trap test; gate goldens + fuzz.

## Phase 2.5 — Worker shells (profile-gated)
8. Workers hold patch-local block, not global pool; per-patch rows. If shells
   dominate: alias patch row; else synced scalars. Measure churn (remove_outside).

## Phase 3 — Geometry pool (last)
9. geom_x/geom_y (N,Vmax) + norm_x/norm_y (N,Nmax); update_transform writes a row.
10. collisions circle-poly: geom_x.take(idx,0)/geom_y.take(idx,0) -> two (K,Vmax),
    no deinterleave; pad by replicating last vertex; parity fuzz bit-exact.
11. Cleanup + editor-lens; refresh docstrings + README bench stats; flake8 rc=0.

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
