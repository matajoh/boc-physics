# XPBD solver migration — single-mode replan (40-draft)

Synthesis of the speed / usability / conservative lenses and five adversarial
rounds, **re-scoped by the user's U1/U2/U3 decisions**: the engine drops the
multi-mode concept entirely and becomes a single XPBD rigid-body solver (full
rotation + friction), exposed through three execution strategies kept as a
teaching comparison — **serial**, **batched** (single-core SoA hardware
parallelism), and **BOC** (multi-core sub-interpreter concurrency).

Every step is green except the explicitly enumerated S2→S3 golden-recapture
window (you cannot recapture serial goldens before the solver is wired), so each
commit stays revertable. The user makes all git commits.

This is a plan only — no production code is written here.

---

## What changed from the modes-kept draft (archived `41-draft-plan-modes-kept.md`)

- **U3 — drop the entire mode concept.** No `PhysicsMode`, no NONE/BASIC/ROTATION,
  no `apply_collision`/`apply_none`/`is_contact_mode`. One physics path: XPBD
  rotation + friction. This *removes* all the NONE-keeping scaffolding the prior
  draft threaded (KEPT-forever lists, the S4.2 NONE seam else-branch, the
  NONE+parallel smoke test, the per-mode engine/parallel/kernel branches).
- **U1 — keep `batched` as a teaching axis, ported to XPBD.** The current batched
  kernel (`kernel.resolve_batched`) is a vectorised *impulse* solver; a fair
  *serial vs batched vs BOC* comparison requires all three to solve the **same**
  XPBD physics. So a new **S3.5** ports the batched path to a colour-batched XPBD
  position+velocity kernel. `--batched` / `use_batched_solver` / `SolveConfig`
  survive; only the inner solve changes.
- **U2 — a single engine `num_substeps` knob** (no mode-specific default), chosen
  after S3 serial validation.

### New architecture: one physics, three execution strategies

| Strategy | Selector | Mechanism | Teaching point |
|---|---|---|---|
| serial | default | `xpbd.solve_group_substep` sequential loop | the reference |
| batched | `--batched` / `use_batched_solver` | colour-batched XPBD SoA kernel, one core | hardware data-parallelism |
| BOC parallel | `--parallel` | sub-interpreter workers over partitioned cowns | multi-core concurrency |

All three call the **same** `xpbd.py` free functions over the same contact data;
they differ only in how work is scheduled. "Same core, different scheduler."

---

## Locked user decisions (non-negotiable, threaded through every step)

- **REPLACE the impulse solver outright** — no A/B runtime toggle between impulse
  and XPBD. `git revert <commit>` is the only backout lever, so every step stays a
  clean, self-contained commit.
- **One physics path (U3).** The `PhysicsMode` enum and every per-mode branch are
  removed (S6 sweep). The three execution strategies are *not* modes — they pick a
  scheduler, not a physics.
- **`num_substeps` is a single engine knob (U2),** its default decided ONLY after
  S3 full-scene serial validation.
- **`batched` stays, ported to XPBD (U1)** at S3.5; the serial/batched/BOC triad
  must solve identical physics for the comparison to be valid.
- **BOC rules:** every XPBD function and `ContactConstraint` is module-level and
  worker-importable; first N params bind the N cowns; extras are trailing
  defaults; no closures over free vars; no threading/sleep/atomics/polling;
  sequencing via the cown graph + per-cown FIFO.
- **Comments ≤ 1 line, ≤ 120 chars.** flake8 `src test bench` rc 0. Python 3.14,
  venv `.env314`. Files double as lecture notes — readability is first-class.
- **Clean rejection-sampled spawns only** in every validation (never reintroduce
  overlapping spawns — that was the KE-explosion artifact, not physics).

---

## The four design decisions (carried over, justification trimmed to one-liners)

### D1 — `module-home`: dedicated `src/bocphysics/xpbd.py`, switch at the scheduler boundary. ✅
XPBD lives in its own module whose docstring cites Müller et al. 2020 Algorithm 2;
`integrate_block` stays in `solver.py` as the shared, solver-agnostic integrator
so `xpbd → solver` is strictly one-way (no cycle). The switch from impulse to XPBD
happens where the scheduler already chooses behavior (`engine.solve_substep`, the
two `parallel` behaviors, the batched kernel entry) — one edited call site each,
not a second solver crammed into `solver.py`'s namespace.

### D2 — `deletion-timing`: delete-on-orphan, but the impulse solver is one big orphan after S4. ✅
Under U3 the impulse solver is shared by all three strategies, so it stays live
until the **last** strategy is ported: serial at S2, batched at S3.5, parallel at
S4. Once all three run XPBD the entire impulse solver + mode concept is orphaned
**together** and removed in one reviewable S6 sweep, gated by a hard
`vscode_listCodeUsages == 0` check on every symbol (see the pre-deletion gate).
Early per-stage deletions are limited to *tests* that a wired strategy breaks
(retired/recaptured in place).

### D3 — `contact-record`: immutable `ContactConstraint` NamedTuple; `solve_positions` RETURNS the per-contact `lambda_n`; do NOT cache `w_n`. ✅
The container-pickle trap only bites a NamedTuple used as a *cown value*;
`ContactConstraint` is a per-substep behavior-local that never crosses a cown.
Returning the lambdas makes the position→velocity coupling (`f_n = lambda_n / h²`)
a visible value, not a hidden field write. Hoist `w_n` only if S3 profiling shows
the velocity-pass `generalized_inverse_mass` is a measurable line item.

### D4 — `helper-idioms`: reuse engine idioms behind a hard VERIFY gate; S1 prototype-parity locks equivalence. ✅
`Matrix.cross()`, `Matrix.perpendicular()`, `scaled_add(s, x, in_place=True)`,
`Matrix.magnitude()`, and `body.move(delta)` exist and are used in
`physics.py`/`contacts.py`. `body.rotate(delta)` is **unconfirmed** — if only
`rotate_to(angle)` exists, use `body.rotate_to(angle + delta)` (S1.0 gate).

Two load-bearing correctness facts (from repo memory):
- **`+=` / snapshot ALIAS TRAP:** `v += other` mutates in place and returns the
  same object; `snapshot_poses` must store **scalar** `(x, y, angle)` and rebuild
  `Matrix.vector([x, y])` on derive — never alias a live pose Matrix.
- **`scaled_add` grouping:** `y + s*x == y.scaled_add(s, x)` is bit-exact only if
  grouping is preserved (do not refold a sub-expression) or goldens drift. **The
  same caveat applies to the position-pass `body.move`/`body.rotate` updates** —
  the S3.5 batched SoA position solve must preserve the serial solve's operand
  grouping per row or the per-colour bit-exactness (S3.5.4) fails.

---

## Verified call graph (grounds the S6 sweep)

The impulse solver and the mode machinery form one connected component, reached
from exactly three entry points (the three strategies):

- serial loop: `engine.solve_substep → solver.solve_group_substep →
  resolve_manifolds → resolve_pair_list →` (FRICTION branch / generic loop) `→
  physics.apply_*`.
- serial batched: same, but `resolve_pair_list → kernel.resolve_batched` when
  `use_batched_solver`.
- parallel: `parallel.solve_intra/boundary_substep → solver.resolve_manifolds → …`,
  and **also `→ kernel.resolve_batched`** when `--batched --parallel` (the worker
  threads `batched=config.batched` into `resolve_manifolds`). So the impulse
  batched kernel has TWO live callers — serial-batched and parallel-batched — and
  is not orphaned until BOTH are ported (serial-batched at S3.5, parallel at S4).

Removed together at S6 once all three are XPBD: `PhysicsMode`, `is_contact_mode`,
`Constraint`/`PreparedContact`, `resolve_collision`, `prepare_collision`,
`prepare_contacts`, `apply_collision`, `apply_none`/`apply_basic`/`apply_rotation`/
`apply_friction`, `restitution_for`, `restitution_bias`, `solve_normal_impulses`,
`apply_accumulated`, `accumulated_friction`, `scatter_impulses`, `TangentData`,
`constraint_height`, `build_tangent_data`, `resolve_manifolds`, `separate_manifold`,
`build_manifold`, `build_group_manifolds`, `resolve_pair_list`,
`solver.solve_group_substep` (impulse), the whole
`kernel.py` impulse kernel, `num_velocity_iterations`, `contacts.separate`
(reached only via `separate_manifold`).

**Kept (XPBD reuses):** `integrate_block` (solver), `contacts.find_contact_points`,
`detect_collision` / SAT, and **only the genuinely solver-agnostic colour pieces**
`greedy_edge_color` (int-pair input) + `pack_bodies` (body-list input), plus a new
`colour_contacts` / constraint-row packer S3.5 adds. **NOT reusable** (they unpack
the impulse `Manifold` 5-tuple or build impulse `denom`/`v_target`):
`colour_manifolds`, `body_rows`, `pack_contacts`, `normal_kernel`, `friction_kernel`
— impulse-only, removed at S6. Also kept: `transport` / `quadtree` / `bodies` /
`geometry`. `contacts.py` keeps `find_contact_points` and loses `separate`.

> Each symbol above is deleted only after `vscode_listCodeUsages` returns zero
> callers in the same commit — the call graph here is the *expectation*, the gate
> is the *proof*.

---

## Numbered implementation plan

### S0 — Baseline (no code)
- **What:** Run `pytest -q` and `flake8 src test bench` on `.env314`; record the
  pass count (context: 810) and rc 0 in this plan dir. Run the current-solver
  serial sweep (`.copilot/pen_probe.py`), the parallel timing skeleton
  (`.copilot/parallel_timing_probe.py`), the **batched** single-core sweep
  (`--batched`), and the prototype sweep (`.copilot/xpbd_probe.py`) into one
  results table — three strategy baselines so S6 can show the post-port triad.
- **Rationale:** the recorded ms/frame × dyn_pen grid per strategy is the only way
  to prove the wins and catch a constant-factor regression.
- **Verify:** numbers committed to the plan dir; baseline green.

### S1 — Serial XPBD core module `src/bocphysics/xpbd.py` (pure addition, behavior-neutral)
- **S1.0 (verify gate, factual):** Confirm `Matrix.cross`, `Matrix.perpendicular`,
  `scaled_add`, `Matrix.magnitude()` (a method — needs parens; restitution gate),
  and **confirm or add** `body.rotate(delta)` — if only `rotate_to` exists, use
  `body.rotate_to(angle + delta)` in the position pass.
- **S1.1:** Create `xpbd.py` with a module docstring stating it is *the* rigid-body
  solver, faithful to Müller et al. 2020 Algorithm 2, 2D-specialised. Import
  `integrate_block` from `solver` (one-way).
- **S1.2 — kinematic helpers (engine idioms):**
  `generalized_inverse_mass(body, r, normal)` (`inv_mass + (r.cross(normal))² *
  inv_inertia`, `0.0` for statics); `contact_velocity(body, r)`
  (`linear_velocity + angular_velocity * r.perpendicular()`, zero for statics);
  `relative_normal_velocity(a, b, r_a, r_b, normal)`.
- **S1.3 — `ContactConstraint` NamedTuple + `build_contacts(pairs, contacts=None)`:**
  immutable `(a, b, normal, r_a, r_b, depth, bias_velocity)`; loop pairs, skip
  static-static (denom guard), run `detect_collision` + `find_contact_points`,
  emit ONE constraint **only for a real (penetrating, `depth > 0`) contact point**.
  `bias_velocity` is the **raw** `relative_normal_velocity` at build time (the
  prototype's `vbar_n`) — the XPBD path's only restitution gate is the `2*g*h`
  test in `solve_velocities`. When `contacts` is given, record points into it for
  the show-contacts overlay (replacing the `build_manifold` overlay hook).
- **S1.4 — position pass:** `apply_positional_impulse(a, b, r_a, r_b, impulse)`
  via `body.move(...)` + `body.rotate(...)`/`rotate_to`;
  `solve_positions(constraints) -> list[float]` — one Gauss-Seidel pass, α=0,
  returning one `lambda_n` per constraint in input order (every constraint
  penetrates, so the list is index-aligned with `constraints` — no zero-fill).
- **S1.5 — velocity halves:**
  `snapshot_poses(bodies)` storing **scalar** `(x, y, angle)` (alias-trap safe);
  `derive_velocities(bodies, prev, h)` (`v=(x−x_prev)/h`, `ω=(angle−angle_prev)/h`,
  rebuilding `Matrix.vector`); `apply_velocity_impulse(...)` mirroring
  `scatter_impulses`; `solve_velocities(physics, constraints, lambdas, h, gravity)`
  — one pass, dynamic Coulomb friction (Eqn 30, `f_n = lambda_n / h²`) then
  restitution (Eqn 34, gate `e = 0 if |vn| <= 2*g*h`, `g = gravity.magnitude()`).
- **S1.6:** `solve_substep(physics, bodies, pairs, gravity, sub_dt, contacts=None)`
  — the six-line centrepiece (snapshot → integrate_block → build_contacts →
  solve_positions → derive_velocities → solve_velocities); and
  `solve_group_substep(physics, bodies, pairs, gravity, sub_dt, num_substeps,
  contacts=None)` = the substep loop. Split so the parallel intra behavior calls
  `solve_substep` once per scheduled substep.
- **S1.7 — `test/test_xpbd.py` (semantics + parity):** prototype-parity on a 2–3
  box stack over several substeps (tight tolerance vs `.copilot/xpbd_probe.py` —
  the equivalence lock for D4); plus semantic units (static→0 inverse mass; `r∥n`
  → `inv_mass`; mass-weighted separation; dynamic-static moves only the dynamic;
  two constraints for a box flat on a box, none for static-static; overlay set
  populated; `derive_velocities` recovers `(x−x_prev)/h`; Coulomb cone cap;
  restitution gated at `2*g*h`; a separating pair yields no constraint and no
  velocity change).
- **S1.8 — gate:** `pytest test/test_xpbd.py` green; `flake8` clean; full suite
  still at the S0 pass count (pure addition, nothing wired).

### S2 — Wire the serial engine to XPBD (serial non-batched path only)
- **S2.1:** In `engine.PhysicsEngine.solve_substep`, branch on the **strategy**:
  `use_batched_solver → solver.solve_group_substep(...)` (the impulse batched path,
  untouched until S3.5), `else → xpbd.solve_group_substep(...)`. Keep the
  signature stable; pass the show-contacts overlay set straight through to `xpbd`.
  One velocity pass only — no `num_velocity_iterations` loop in the XPBD path.
  *Rationale: the serial reference becomes XPBD now; the batched arm stays impulse
  for exactly one stage (ported next at S3.5), so `--batched` keeps running.*
- **S2.2:** Document at the branch: `# --batched still runs the impulse kernel
  until S3.5 ports it to XPBD; the two are NOT yet comparable.`
- **S2.3:** Do **not** change `num_substeps` defaults (deferred to S3 per lock).
- **S2.4:** Repoint the core-match tests (`test_solver_core_matches_engine_substep`,
  `test_polygon_group_core_matches_engine`) candidate side from
  `solver.solve_group_substep` to `xpbd.solve_group_substep` so they stay genuine
  engine→core forwarding checks (the impulse `solver` core no longer matches the
  XPBD engine path). *Fixes the relabel-without-repoint red.*
- **S2.5 — gate:** `pytest test/test_xpbd.py test/test_solver.py` green; **run the
  FULL suite and enumerate the actual red set** as the deliverable. Known reds to
  expect: the serial XPBD integration goldens that route through `engine.step` in
  `test_engine` and `test_solver` (the same two files S3.2 recaptures), and the
  cross-strategy settle tests — `test_parallel_settles_like_serial` (L445),
  `test_slab_stepper_settles_like_serial` (L487),
  `test_quadtree_fallback_settles_like_serial` (L576) — whose serial side is now
  XPBD while parallel stays impulse until S4. **xfail those three with a
  `# cross-solver window: re-unified at S4` note** (removed and re-asserted at
  S4.6). `flake8` clean. `git diff --stat` should show `engine.py` (+strategy
  branch), `xpbd.py`, the repointed/relabelled tests, the three xfail markers —
  nothing else. *Reverting S2 restores the serial impulse path.*

### S3 — Full-scene serial validation + retire serial-specific tests + pick `num_substeps`
- **S3.1:** Retire the serial impulse-only characterization tests that the XPBD
  serial path no longer satisfies (e.g. serial penetration magic-number goldens),
  recapturing intent as invariants in S3.2. No production-symbol deletion — the
  impulse solver is still batched-live (S3.5) and parallel-live (S4).
- **S3.2:** Recapture genuinely-changed serial goldens in `test_engine.py` /
  `test_solver.py`, **preferring physical invariants** (penetration below tol, KE
  bounded, rests-above-floor, left/right symmetry) over re-pinned magic numbers;
  reserve exact-equality for determinism / same-core parity. Each rewritten
  assertion must still fail on a deliberately broken solver.
- **S3.3:** Full-scene serial validation with the clean rejection-sampled bench
  (`bench/drop_box.py`, never overlapping spawns): confirm dynamic penetration in
  the prototype band and bounded KE across the substep grid; produce the
  integrated-engine ms/frame × dyn_pen curve vs the S0 serial baseline.
- **S3.4 (user decision — U2):** Pick the **single** `num_substeps` value from
  S3.3 at the convergence knee (prototype: 4→1.02, 6→0.19, 8→0.09 dyn_pen).
  **Do NOT change the `PhysicsEngine.num_substeps` default value here** — at S3 the
  batched and parallel paths still run impulse, so bumping the shared default would
  shift their goldens. Instead: (1) `grep -rn num_substeps src test bench` and audit
  every reader; (2) pass `num_substeps` **explicitly** in bench/probe/tutorial
  scenes; (3) defer the engine default-value change to **S6** (all paths XPBD by
  then). Note `test_parallel` uses module constant `NUM_SUBSTEPS=4` (unaffected).
- **S3.5-gate:** full suite green (scenes pin `num_substeps` explicitly, engine
  default unchanged); `flake8 src test bench` rc 0. **Serial XPBD is stable,
  non-penetrating, clean — the gate before porting the other two strategies.**

### S3.5 — Port the batched strategy to XPBD (NEW; U1 — keep the teaching comparison)
- **S3.5.1 — colour over constraints + file location:** Re-point only the reusable
  colour pieces (`greedy_edge_color`, `pack_bodies`) at `xpbd.ContactConstraint`s and
  add `colour_contacts(constraints)` + a constraint-row packer — same graph (two
  constraints sharing a movable body conflict), same deterministic input-order
  colouring (worker-count independent). `colour_manifolds`/`body_rows`/`pack_contacts`
  are impulse-shaped (they unpack the `Manifold` 5-tuple / build `denom`/`v_target`)
  and are NOT reused. **Decide here** whether the XPBD kernel lives in a new
  `xpbd_kernel.py` (cleanest — lets S6 delete the impulse functions in `kernel.py`
  while keeping the migrated colour helpers) or alongside the impulse code; the S6
  granularity (below) depends on this choice.
- **S3.5.2 — batched XPBD kernel** (file per S3.5.1):
  per substep, snapshot → `integrate_block` → for each colour run a **batched
  position solve** (one colour = body-disjoint constraints, so the SoA Jacobi
  update equals the sequential `solve_positions` over that colour exactly — note
  this holds for the POSITION pass too: disjoint bodies means no constraint in the
  colour reads a pose another constraint in the same colour just moved, so there
  is no mid-colour ordering dependency) → `derive_velocities` → for each colour a
  **batched velocity solve** (friction + restitution as row-wise Matrix ops with
  scatter-add). Within a colour bodies are disjoint so batched == serial
  bit-for-bit; across colours it is Gauss-Seidel in colour order — gated by
  settling-band tests, not bit-exact goldens (same contract the impulse batched
  kernel had).
- **S3.5.3 — wire the strategy:** make the S2.1 `use_batched_solver` arm call the
  XPBD batched kernel instead of `solver.solve_group_substep`; remove the S2.2
  "not yet comparable" caveat. `--batched` now runs XPBD.
- **S3.5.4 — tests:** `test/test_xpbd_kernel.py` — per-colour batched-vs-serial
  bit-exactness on a single colour; full-scene settling-band parity vs serial XPBD
  at the S3.4 substep count; determinism (worker-count independent colouring →
  identical result). Repoint/retire the `test_kernel.py` tests that characterise
  the *serial-batched* impulse path. **The impulse `kernel.resolve_batched` is NOT
  orphaned here** — the parallel worker still calls it via `batched=config.batched`
  (`--batched --parallel`) until S4, so its body and remaining tests are swept at
  S6 with the rest of the impulse solver.
- **S3.5.5 — gate:** full suite green; `flake8 src test bench` rc 0; a three-way
  ms/frame probe (serial / batched / impulse-parallel) confirms batched-XPBD is a
  real single-core speedup over serial-XPBD at matched dyn_pen. *Reverting S3.5
  restores the impulse batched path.*

### S4 — BOC parallel port (Option B) — the last strategy to XPBD
- **S4.1:** In `parallel.solve_intra_substep`, replace the `integrate_block +
  resolve_manifolds` body with `xpbd.solve_substep(config.physics, dyn_shells,
  pairs, config.gravity, config.sub_dt)` (no overlay in workers). `snapshot_poses`
  runs inside `solve_substep`, so `x_prev` is **local to the intra behavior** —
  Option B adds zero state-block columns. No mode branch (one physics). **(If S5.2
  escalates to Option A, this body changes: the intra behavior must persist
  `x_prev` into the state block for the seam to read — so decide A-vs-B BEFORE
  writing S4.1, see S5.2.)** **This
  drops the worker's `batched=config.batched` path:** `--batched --parallel` no
  longer batches *within* a patch (each patch runs the serial XPBD `solve_substep`).
  `--batched` selects the single-core batched kernel for the non-parallel engine;
  combining it with `--parallel` is not a named teaching axis. Removing this call
  site is what finally orphans `kernel.resolve_batched` (swept at S6). **This
  reds `test_intra_behaviour_uses_batched_flag` (test_parallel.py:268)** — it
  asserts the worker (config.batched=True) matches `serial_intra_reference(...,
  batched=True)`; with the worker now non-batched XPBD, retire that test in this
  commit and drop the `batched` argument from `serial_intra_reference`.
- **S4.2:** In `parallel.solve_boundary_substep`, replace `resolve_manifolds(seam)`
  with `constraints = build_contacts(seam_pairs)`, `lambdas =
  solve_positions(constraints)`, **then `solve_velocities(physics, constraints,
  lambdas, sub_dt, gravity)`**. The seam keeps full Coulomb friction + restitution;
  the ONLY omission vs a global XPBD pass is `derive_velocities` on the seam's own
  push — that increment stays sampled from each patch's intra-local `x_prev`.
  Docstring states the asymmetry's direction: the seam under-couples
  position→velocity (no `derive_velocities` at the seam), damping rather than
  injecting energy (conservative); residual bounded by the S5.2 invariants.
- **S4.3:** Repoint `test_parallel.py` seam references (`serial_intra_reference`,
  `serial_boundary_reference`) to the `xpbd` core — `solve_substep` for intra;
  `build_contacts → solve_positions → solve_velocities` for boundary (matching the
  S4.2 seam) — so "worker == serial core" holds with seam-velocity coverage.
  `num_substeps`/`batched` on `SolveConfig` stay **inert** this stage (schema
  removal in S6).
- **S4.4 — deletion-confinement check:** parallel tests green except the seam
  goldens (recaptured in S5), the still-xfailed cross-strategy settle tests
  (removed in S4.5), and `test_intra_behaviour_uses_batched_flag` (retired in S4.1
  with the worker batched-drop); full suite otherwise green; `flake8` rc 0. Edits confined to
  `solve_intra_substep` + `solve_boundary_substep`. *Reverting S4 restores the old
  parallel path.* **No production-symbol deletion yet** — the impulse solver is now
  fully orphaned (serial=S2, batched=S3.5, parallel=S4) and swept in one commit at
  S6, the cleanest reviewable shape for removing a whole subsystem + the mode enum.
- **S4.5 — re-unify the cross-strategy guard (stage gate):** Now all strategies are
  XPBD, so **remove the three `# cross-solver window` xfail markers** added at S2.5
  and assert each PASSES within its `abs=2.0` / `+1.0` tolerance (no silent XPASS).
  **Parametrize all three over `{4, the S3.4-chosen count}`** so the guard does not
  only ever validate substeps=4. Thread a `num_substeps` argument through
  `settle_serial`/`settle_parallel*` into their `PhysicsEngine(...)` calls. This is
  the real S4 gate: the most important parallel regression guard live and green
  before S5.

### S5 — Parallel accuracy go/no-go + seam goldens (Option B vs Option A)
- **S5.1:** Recapture `test_seam_decomposition_*` for the position-only XPBD seam as
  a **tolerance band** (parallel-XPBD penetration within X of serial-XPBD), not
  bit-exact goldens.
- **S5.2 — concrete numeric go/no-go (accuracy, NOT throughput):** Measure
  serial-XPBD vs parallel-XPBD dynamic penetration and settled KE at the **chosen
  S3.4 substep count, passed explicitly**, on (a) a seam-crossing scene and (b) a
  partitioned settled pile, clean rejection-sampled.
  - **PASS — ship Option B:** parallel `dyn_pen ≤ serial dyn_pen + max(0.05, 0.5 ×
    serial dyn_pen)` on **both** scenes **AND** parallel settled KE within the same
    order of magnitude as serial (bounded, no growth with substeps) **AND** two
    seam-physics invariants `dyn_pen`+KE are blind to: **(i) tangential/shear** — a
    body on a seam-straddling slope must not slide further than serial-XPBD by more
    than a tolerance (catches lost seam friction); **(ii) restitution** — a body
    dropped onto a seam-straddling surface bounces within tolerance of serial-XPBD.
  - **FAIL — escalate to Option A:** widen the `(N×7)` state block to `(N×10)`
    carrying `x_prev` (2) + `angle_prev` (1); the seam re-derives `v=(x−x_prev)/h`
    after its position correction. A Matrix block crosses by pointer regardless of
    width, so the only added cost is 3 scalars/row in `apply_state`/`store_state` —
    A vs B is an *accuracy* decision, not a speed one. **A-escalation ripple:**
    widening changes `pack_state`/`apply_state`/`store_state` column layout, which
    `test_transport` pins and the S4.3 bit-exact references depend on — recapture
    `test_transport`, the S4.3 references, and any `transport.*` column-index
    constants in the SAME A commit. **Decide A-vs-B BEFORE S4.1** (run a quick
    S5.2-style seam probe up front): an A decision reworks the bodies themselves —
    S4.1 (intra persists `x_prev` to the block) and S4.2 (seam re-derives
    `v=(x−x_prev)/h` from block `x_prev`), not just the transport layout and the
    S4.3 references.
- **S5.3 — gate:** full suite green, **including `test_worker_count` re-run** to
  confirm parallel-XPBD stays worker-count deterministic (colour-ordered seam);
  `flake8` rc 0. Parallel done; all three strategies now solve identical XPBD.

### S6 — Remove the mode concept + dead impulse solver; benchmarks; docs
- **S6.0 (retire the impulse test surface FIRST, same PR, before the gate):** The
  impulse `physics.py` methods stay live until this stage, so `test_physics.py`
  (`Physics(PhysicsMode.FRICTION)`, `apply_friction`, `solve_normal_impulses`,
  `Constraint(PhysicsMode.FRICTION, ...)` ~L245), the `Constraint`-constructing
  test in `test_solver.py` (~L210), and any remaining impulse-only `test_kernel.py`
  tests are GREEN through S2–S5 and reference symbols S6.1 deletes. Retire or
  repoint them in the same commit so `vscode_listCodeUsages` counts no test
  callers. *The hard gate below is meaningful only after the test surface is gone.*
- **S6.1 (pre-deletion gate, HARD):** With all three strategies on XPBD and the
  impulse tests retired (S6.0), run
  `vscode_listCodeUsages` on every symbol in the "Verified call graph" removal list
  and confirm **zero remaining callers** (each call site is itself being removed in
  this commit). Then delete, in one focused commit: the `PhysicsMode` enum +
  `is_contact_mode` (config.py); `Constraint`/`PreparedContact`, `resolve_collision`,
  `prepare_collision`, `prepare_contacts`, `apply_collision`, `apply_none`,
  `apply_friction`, `restitution_for`, `restitution_bias`, `solve_normal_impulses`,
  `apply_accumulated`, `accumulated_friction`, `scatter_impulses`, `TangentData`,
  `constraint_height`, `build_tangent_data`, `ZERO_VEC` (physics.py); `resolve_manifolds`,
`separate_manifold`, `build_manifold`, `build_group_manifolds`, `resolve_pair_list`,
`solve_group_substep`, the `Manifold`/`ContactSet` aliases (solver.py); the impulse
`kernel.py` functions (`colour_manifolds`, `body_rows`, `pack_contacts`,
`normal_kernel`, `friction_kernel`, `resolve_batched` — keeping `greedy_edge_color`/
`pack_bodies` if S3.5.1 left the XPBD kernel in `kernel.py`, else delete the file);
`contacts.separate`;
  `num_velocity_iterations` from `engine`/`parallel`/`SolveConfig`/seeds; and the
  `--mode` CLI arg + `PhysicsMode` imports across `__init__.py`/`simulation.py`/
  `bench`. **NamedTuple/constructor surgery (focus: positional ripple):** drop the
  `mode` (first field) and `restitution_threshold` fields from the `Physics`
  NamedTuple ([physics.py:79](src/bocphysics/physics.py#L79)); drop `mode` from
  `PhysicsEngine.__init__` ([engine.py:40](src/bocphysics/engine.py#L40)); and
  update EVERY positional construction — `Physics(PhysicsMode.FRICTION)` and
  `PhysicsEngine(w, h, PhysicsMode.FRICTION, ...)` — across test_engine,
  test_parallel, test_kernel, test_worker_count, test_contacts, test_solver,
  test_physics, `simulation.py`, and benches. Note `Physics` rides the noticeboard
  as a cross-worker snapshot, so this is also a config-schema change. Keep
  `integrate_block`, `find_contact_points`, `detect_collision`, the migrated colour
  helpers, and `transport`/`quadtree`/`bodies`/`geometry`. *Rationale (D2):
  removing a whole orphaned subsystem + an enum is the higher-risk, wider-blast
  edit — batch it into one reviewable, revertable commit behind the usage gate.*
- **S6.2:** Now change the `PhysicsEngine.num_substeps` default to the S3.4 value
  (all paths XPBD; the deferred default change from S3.4) and recapture any
  default-dependent goldens in this commit.
- **S6.3:** Run `bench/drop_box.py` (+ stack/tutorial) `--runs 5`; publish the
  three-strategy (serial / batched / BOC) ms/frame × dyn_pen grids vs the S0
  baseline; update README tables and drop the `--mode` row.
- **S6.4:** Update `docs/concepts/parallel-solver.md` and `docs/tutorial/04-*`/`05-*`
  to describe the single XPBD substep, the three execution strategies, and the
  position-only seam; fix the references doc (add Catto 2014 TGS-soft; correct the
  Macklin-2019 "4 substeps enough" note to the measured 6–8 knee). Remove
  multi-mode pedagogy prose.
- **S6.5 — gate:** final `pytest` + `flake8 src test bench` rc 0; run the
  `finalize-pr` editor-lens comment sweep over the diff.

---

## Known trade-offs / risks

1. **Batched-XPBD kernel is genuinely new code (S3.5).** The colour graph and SoA
   packers are reused, but the per-colour XPBD position+velocity solve is a fresh
   kernel; its correctness rides the per-colour bit-exactness test (disjoint bodies
   ⇒ Jacobi == serial) plus a full-scene settling band. If the SoA position solve
   proves fiddly, a fallback is a *non-vectorised* per-colour loop that still
   demonstrates the colouring (slower, but keeps the teaching comparison) — surfaced
   as a sub-decision if S3.5.2 stalls.
2. **Parallel seam accuracy (Option B sufficiency)** is the one unmeasured unknown;
   the S5.2 numeric gate makes the B→A decision concrete. Option A is scoped.
3. **One big S6 deletion commit** is wider-blast than incremental deletes, but the
   impulse solver only becomes fully orphaned at S4, and a single mode-concept
   removal is exactly the clean "drop it" the user asked for. The hard usage gate
   is the safety net.

---

## Decisions resolved by the user (recorded)

- **U1 — `batched` stays, ported to XPBD** at S3.5 (so serial/batched/BOC compare
  identical physics). *Resolved: in scope for this PR.*
- **U2 — single engine `num_substeps`** (no mode-specific default), value chosen at
  S3.4, default changed at S6.2. *Resolved.*
- **U3 — drop the entire mode concept.** One physics path (XPBD rotation+friction);
  `PhysicsMode`/NONE and the impulse solver removed in the S6 sweep. *Resolved.*

No open user decisions remain. Implementation (S0→S6) is gated on approval of this
plan.
