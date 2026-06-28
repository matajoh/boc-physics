# Narrow-phase optimisation — PLAN

Status: **PLAN DRAFTED. Do NOT implement until the user approves and says go,
per task.** Drafted 2026-06-28.

Origin: speculative Q&A 2026-06-27. Narrow phase = ~73% of compute (confirmed by
serial cProfile: `build_contacts` cumtime 3.60s / 5.11s = 70.5%;
`detect_collision` 2.04s, contact-point scan ~0.93s+).

## Goal & scope

Cut the narrow phase (currently ~73%) on **both** the serial and parallel paths.
Three approved tasks:

- **A** — AABB cheap reject
- **B** — vectorise the contact-point scan
- **C** — batched SAT across pairs

**D** (per-substep contact caching / skip re-detect) is **REJECTED** by the user.

Each task is independently shippable; land A → B → C with a checkpoint each.

## Decisions taken

- Slabs **are vertical** (x-axis cut, aligned with gravity) — confirmed:
  `build_slab_partition` sorts by `position.x`. The earlier "dense pile bottom"
  framing was wrong; a vertical slab owns a full top-to-bottom column, so the
  load imbalance is horizontal.
- **A** — cheap reject uses `AABB.disjoint`, **not** a bounding circle. The AABB
  is already cached per body (`bodies.py` `.aabb` lazy property), `disjoint` is 4
  float compares, and it is tighter than the conservative max-vertex-length
  radius for rotated boxes. APPROVED.
- **B** — vectorise the contact-point scan. APPROVED ("no-brainer").
- **C** — batched SAT across pairs. APPROVED (bigger lift).
- **D** — rejected.

## The hot path (where the ~73% lives)

`xpbd.build_contacts` (`xpbd.py`), called once per `solve_substep`, 8×/frame,
per pair in a Python loop. Two costs per pair:

1. `collisions.detect_collision` (SAT) — ~57% of `build_contacts`.
   - `intersect_polygon_polygon` (`collisions.py`): concat normals, transpose,
     two matmuls (`a_proj`, `b_proj`), 4 axis min/max, `less` ×2, `where`,
     `argmin`, index, `vecdot`, `min`. ~15 Matrix ops/pair, all per-pair Python
     dispatch.
   - `intersect_circle_polygon`: plus `closest_vertex_on_polygon`, a per-vertex
     Python min loop.
2. `contacts.find_contact_points` — ~26%.
   - `find_contact_points_polygon_polygon` → `edge_point_distances` (builds the
     (E×P) distance block in C) **then** `scan_polygon_edges` tears it apart
     scalar-by-scalar (`row = [distances[i, j] for j in ...]`) and
     `scan_edge_points` runs a Python double loop. Defeats the C block it just
     built.

Parallel path (`parallel.py`) calls the **same** `build_contacts` inside each
intra behavior (`solve_intra_substep` → `xpbd.solve_substep`) and inside each
seam (`solve_boundary_substep` → `xpbd.build_contacts` directly). So the narrow
phase is **replicated** across patches + seams, not reduced. Speeding it up helps
both paths. The batched kernels (`kernel.py` / `xpbd_kernel.py`) already batch the
**solve**; the narrow phase is the last un-batched per-pair Python loop.

## Step 0 — Baseline (RECORDED 2026-06-28)

Canonical profiling harness: `.copilot/profile_narrow.py` (streams shapes in
waves to a dense settled pile, then cProfiles a measured window and reports
ms/frame + build_contacts share). Standard scene = **~200 bodies** (denser than
the original 80): the narrow-phase *share* is scene-insensitive (~71% @80,
~69% @200 — solve and broad phase scale with contacts too), but the larger
absolute per-frame cost gives a cleaner ms signal, and Task C's batching win
*grows* with pair count, so a dense scene avoids understating it.

Baseline numbers:
- Tests: **824 passed**, no pre-existing failures. `flake8 src test bench` rc=0.
- Profile (`.copilot/profile_narrow.py --target 200 --measure 120`):
  - seed 7: 135.8 ms/frame, build_contacts 69.5%
  - seed 8: 136.7 ms/frame, build_contacts 70.0%
  - seed 9: 124.1 ms/frame, build_contacts 68.2%
- Re-measure after each task with the SAME command + seeds 7/8/9.

## Validation backbone (applies to EVERY task)

- **Parity-oracle fuzz** — established precedent: `test_collisions.py` already
  does this for the per-axis → batched SAT. For each change: embed/retain a
  reference function reproducing the pre-change behaviour, fuzz random poses,
  assert **bit-exact** (allclose only if an fma is introduced).
  `test_collisions.ref_intersect_*` and `test_contacts.overlapping_box_pair` are
  the model.
- **Feature-ID + contact-point stability** already guarded by `test_contacts.py`
  (`test_feature_id_*`, bit-exact point equality). Task B must keep these green.
- **Settling band / goldens**: `test_parallel`, `test_xpbd`, `test_engine`.
- Re-profile + re-bench after each task; confirm the narrow-phase % actually
  drops.
- NOTE: the SAT is **already batched across axes** (one matmul, per the
  `test_collisions.py` docstring). Task C adds the **across-pairs** dimension; it
  is not redoing the per-axis batching.

## Task A — AABB cheap reject (low risk, no physics change)

- In the `build_contacts` loop, before `detect_collision`:
  `if a.aabb.disjoint(b.aabb): continue`.
- Saves the ~15-op SAT body for separated candidates (the swept broad phase is
  frame-level; many candidates are not touching at a given substep).
- `update_transform` cost is paid anyway (the SAT needs `transformed_vertices`),
  so the reject is ~free (4 compares); the AABB is computed inside
  `update_transform`.
- CAVEAT: `build_contacts` already skips when `collision is None`; the AABB reject
  just short-circuits earlier. It must **not** change which contacts emit —
  `AABB.disjoint` is conservative (never rejects a real overlap), so the emitted
  set is identical and bit-exact goldens should hold. Verify against goldens +
  settling-band tests anyway.
- Batched variant later: precompute all candidate AABBs as blocks and mask
  disjoint pairs in one pass. The scalar reject is the first, trivial step.

### Task A — DONE (2026-06-28, bit-exact)

Added `if a.aabb.disjoint(b.aabb): continue` before `detect_collision` in
`build_contacts`. All 824 tests pass (goldens prove bit-exact), flake8 clean.
Measured (`--target 200 --measure 120`, vs baseline):
- seed 7: 135.8 → 126.6 ms/frame (−6.8%), share 69.5% → 67.3%
- seed 8: 136.7 → 126.8 ms/frame (−7.2%), share 70.0% → 67.6%
- seed 9: 124.1 → 117.4 ms/frame (−5.4%), share 68.2% → 66.5%

## Task B — vectorise contact-point scan (low risk)

### Status (2026-06-28)

- **Part 1 DONE — `closest_vertex_on_polygon` (no upstream):** replaced the
  per-vertex Python min loop with
  `verts[(verts - point).magnitude_squared(axis=1).argmin()]`. Bit-exact:
  `magnitude_squared` is sign-independent (so `verts - point` matches the old
  `point - v`), and `argmin` returns the first min like the old strict `<`.
  Added `test_closest_vertex_matches_reference` (2000-pose fuzz vs an embedded
  per-vertex oracle). 825 tests pass, flake8 clean. Re-profile (vs post-A):
  seed7 126.6→124.95 (−1.3%), seed8 126.8→125.15 (−1.3%), seed9 117.4→117.01
  (−0.3%). Small because this path only fires on circle-vs-polygon pairs and
  the canonical scene is box-heavy.
- **Part 2 SUPERSEDED — approximate vectorised fold, folded into Task C:** the
  `scan_polygon_edges` / `scan_edge_points` fold is sequential (running tol-reset
  threshold) so it cannot be reduced bit-exactly. Relaxing exactness, it CAN be
  vectorised approximately — prototyped in `.copilot/proto_contacts.py`:
  per-vertex `min(axis=0)` over `edge_point_distances`, `concat` both passes,
  `argmin` for `c0`, masked `argmin` (within `tol` AND `are_different`) for `c1`.
  Divergence vs the sequential oracle over 13 896 overlapping pairs: manifold
  SET differs 0 %, cardinality flips 0, ONLY order-only `c0`/`c1` swaps 0.09 %
  (two equidistant corners on a flat edge — a relabelling, not a different
  contact). VERDICT: per-pair timing is only 1.04× (a wash) — small polygons
  (3–6 verts) trade a short Python loop for ~10 Matrix C-calls. So do NOT ship
  per-pair (it would forfeit bit-exact goldens for ~0 % gain). The win needs
  batching across pairs, which is Task C. The validated formulation + divergence
  probe in `.copilot/proto_contacts.py` is the proven building block for C, so
  C is substantially de-risked: the contact-localisation math and its
  approximation margin are already characterised.
- (Upstream scalar-iterator ask is now moot for this path — vectorising the fold
  supersedes the bulk-extraction need. Logged separately as a generic nicety.)

- Replace the `scan_polygon_edges` / `scan_edge_points` scalar loop with Matrix
  reductions over the (E×P) `edge_point_distances` block: `min` over the block
  for the closest point, `argmin` for the feature index, second-min within `tol`
  for the 2-point manifold.
- Also `closest_vertex_on_polygon` (`collisions.py`): replace the per-vertex
  Python min with `(point - verts).magnitude_squared(axis=1).argmin()`.
- TRAP: the two-point manifold selection in `scan_edge_points` has a specific
  sequential tie-break (`d < min - tol` resets; within `+tol` and
  `are_different` adds the 2nd point). It must be reproduced **exactly** or
  contact points shift and goldens break. Keep the exact selection order;
  validate `find_contact_points` outputs bit-exact on fixed poses **before**
  wiring into the solve. Feature IDs `(source_uid, vertex_index)` must match.

## Task C — batched SAT across pairs (biggest win, more work)

- DE-RISKED: the contact-point localisation half is already prototyped and
  validated — `.copilot/proto_contacts.py` vectorises `find_contact_points_
  polygon_polygon` and proves it manifold-set-identical to the sequential oracle
  (only 0.09 % order-only c0/c1 swaps). C's remaining novelty is the batched SAT
  and the ragged gather/stack, not the manifold math.
- Same pattern as the colour-batched solve kernel ("same core, different
  scheduler"). Group candidate pairs by type — poly-poly, circle-poly,
  circle-circle — and run the SAT as stacked Matrix ops over all K pairs of a
  type at once, amortising Python dispatch (≈150 pairs → a few calls).
- Per-pair math is already vectorised; the lift is from "1 pair/call" to "K
  pairs/call". Needs ragged handling: polygons have varying vertex/normal counts.
  Pack by `(na, nb)` shape buckets, or pad to max and mask. Edge-colouring is
  **not** needed (detection is read-only, no scatter conflict) — it is purely a
  gather/stack problem.
- Output: per-pair `(normal, depth)` + contact points, scattered back into the
  constraint list `build_contacts` produces. Keep the `ContactConstraint` field
  order and per-point feature IDs so `solve_positions` / `kernel.py` /
  `xpbd_kernel.py` consume it unchanged.
- Validation: bit-exact vs serial `detect_collision` on a fixed pose set across
  many seeds (like the kernel Gate). Then settling-band vs goldens.
- Sequencing: A and B land first (cheap, isolated). C is the structural lift and
  also raises the parallel ceiling (less per-behavior narrow-phase Python).

## Sequencing & checkpoints (each task = its own reviewable changeset)

0. Baseline (above). Record numbers in this file.
1. Task A (AABB reject): edit + green tests + re-profile. CHECKPOINT. review-loop.
2. Task B (vectorise scan): edit + parity fuzz + green tests + re-profile.
   CHECKPOINT. review-loop. (B's vectorised localisation is reused by C.)
3. Task C (batched SAT across pairs): design sub-step first (ragged handling),
   then edit + parity fuzz vs the per-pair oracle + settling band + re-profile.
   CHECKPOINT. review-loop (or branch-review for the whole arc).
4. finalize-pr: editor-lens over the diff, refresh README bench stats, bump the
   version if warranted. **The user commits (the agent never commits).**

STOP after each checkpoint for user sign-off before the next task.

## Risks / open questions (resolve at each task's design step)

- **C ragged vertex counts**: shape-bucket by `(na, nb)` vs pad-to-max + mask.
  The scene is mostly boxes (4v) + regular polys (3–8v) + circles → few buckets.
  Decide in C.
- **bocpy ops for B/C batched reductions**: `argmin` / `min(axis)` exist; the
  second-min for the manifold may need a `min` over a masked block. Confirm
  before coding B.
- **A stale-aabb edge case**: statics never move; dynamics set the dirty bit in
  integrate → `.aabb` recomputes. The conservative test means the set stays
  identical. Confirm there is no edge case.
- **C output** must preserve the `ContactConstraint` field order + per-point
  feature IDs so the downstream solver consumes it unchanged.

## Deferred follow-up (NOT part of this plan)

- `by_uid` per-interpreter memo in `shell_cache` keyed by the patch uid-set,
  evicted on the geometry version: collapses the 8×/frame dict rebuild to
  ≤ worker_count. A micro-opt vs the ~73% narrow phase; revisit only if it shows
  up in a parallel profile. The noticeboard is the **wrong** home (shells are
  interpreter-local; the 64-entry cap; it re-pickles every frame as the partition
  re-cuts).
