# References — Parallel Rigid-Body Contact Solving

This directory collects the primary sources behind the Phase 5.1 / 5.2
parallelisation investigation. The goal of that investigation is to run the
per-frame contact solve across `bocpy` worker sub-interpreters **without**
sacrificing the settling quality of the serial Gauss–Seidel engine, and while
keeping the code readable as a 4M26 teaching artifact.

The papers and slide decks here answer one question three different ways:

> *How do you solve a dense pile of resting contacts in parallel without the
> pile jittering, drifting, or exploding?*

Each source is mapped below to one of three families of answer, and to what we
have already measured in our own probes.

---

## Obtaining the PDFs

The PDFs themselves are **not committed** — they are copyrighted and large
(~39 MB total), so `refs/*.pdf` is gitignored. Download them from the canonical
sources below into this directory. Filenames must match for the prose
references in this README to line up.

| File | Source URL |
|------|------------|
| `ArchitectingJoltPhysics_Rouwe_Jorrit.pdf` | <https://jrouwe.nl/architectingjolt/ArchitectingJoltPhysics_Rouwe_Jorrit.pdf> |
| `ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf` | <https://box2d.org/files/ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf> |
| `Tonge-2012-MassSplittingForJitterFreeParallelRigidBodySimulation-preprint.pdf` | DOI <https://doi.org/10.1145/2185520.2185601> (preprint: <http://www.richardtonge.com/papers/Tonge-2012-MassSplittingForJitterFreeParallelRigidBodySimulation-preprint.pdf>) |
| `smallsteps.pdf` | <http://mmacklin.com/smallsteps.pdf> |
| `XPBD.pdf` | <http://mmacklin.com/xpbd.pdf> |
| `PBDBodies.pdf` | <https://matthias-research.github.io/pages/publications/PBDBodies.pdf> |

Convenience re-fetch (run from inside `refs/`):

```bash
curl -L -o ArchitectingJoltPhysics_Rouwe_Jorrit.pdf \
  https://jrouwe.nl/architectingjolt/ArchitectingJoltPhysics_Rouwe_Jorrit.pdf
curl -L -o ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf \
  https://box2d.org/files/ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf
curl -L -o smallsteps.pdf http://mmacklin.com/smallsteps.pdf
curl -L -o XPBD.pdf       http://mmacklin.com/xpbd.pdf
curl -L -o PBDBodies.pdf  https://matthias-research.github.io/pages/publications/PBDBodies.pdf
# Tonge 2012: access via DOI https://doi.org/10.1145/2185520.2185601
```

---

## 0. Why we need the literature

Our own derisk probes (recorded in `../PLAN.md` and the session plan) reached a
hard wall:

- **Islands collapse.** A settled 80-body open-box pile is **one island**
  (max-island 100 % across all seeds). Island-level task parallelism therefore
  yields width 1 — no speedup. This is precisely the regime Jolt's
  `LargeIslandSplitter` exists to handle (see §1).
- **Spatial decomposition hits a pose-lag wall.** The ghost-row design (Probe E
  / Probe D pose-lag arm) lets each patch resolve its boundary contacts against
  a *prior-substep* snapshot of the neighbour. When each side runs its **own**
  narrow phase against a lagged neighbour pose, the two half-impulses stop being
  equal-and-opposite. Measured result at the locked operating point
  (per-substep, ω = 1.0, seed 7): momentum drift **9242**, penetration **1.32**
  vs the serial reference **0.13** — and under-relaxation (ω = 0.5/0.25/0.1)
  reduces drift but never rescues penetration. Naive parallel Jacobi injects
  energy at the seams.

These two walls are exactly the failure modes the sources below were written to
defeat. The literature splits into three approaches; we intend to build the two
viable ones **the proven way** (probe → gate → spot-review → run → decision) and
let the data choose.

---

## 1. Graph colouring of the constraint graph

**Idea.** Build the constraint graph (nodes = bodies, edges = contacts).
Greedily colour the edges so that no two contacts of the same colour share a
body. All contacts in one colour touch disjoint bodies, so they can be solved in
parallel with no write conflicts. Place a barrier between colours; iterate the
colour sweep. Convergence is *identical to* an unsplit island — only the
evaluation order within an iteration changes.

**Sources.**

- `ArchitectingJoltPhysics_Rouwe_Jorrit.pdf` — Jorrit Rouwé, *Architecting Jolt
  Physics for Horizon Forbidden West*, GDC 2022. Describes the
  `LargeIslandSplitter`: when an island is too large to solve serially, Jolt
  splits it into colour batches (greedy, no two constraints in a batch share a
  body), solves batches in parallel with a barrier between them, and notes the
  result is "almost the same as the unsplit island, only the evaluation order
  changes." Slides/notes: <https://jrouwe.nl/architectingjolt/ArchitectingJoltPhysics_Rouwe_Jorrit_Notes.pdf>;
  source: <https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md>.
  Jolt's splitter cites Chen et al., *High-Performance Physical Simulations on
  Next-Generation Architecture with Many Cores*.
- `ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf` — Erin Catto, *Modeling
  and Solving Constraints*, GDC 2009. The canonical derivation of the
  sequential-impulse / projected Gauss–Seidel contact solver that Box2D, and our
  own engine, are built on. Box2D v3 later adds graph colouring plus 8-wide SIMD
  on top of this solver (<https://box2d.org/>). Read this to understand exactly
  what a "contact" and an "impulse iteration" mean in our solver before
  reordering them.

**Map to our findings.** Phase 5.1 Gate B already **passed**: a coloured sweep
(strict and async) settles a real pile and matches the serial result — colouring
*converges*. What we never actually built was the **coarse** colour-batch
realisation (one behaviour per colour, build-manifold-once-resolve-5×, barrier
between colours). Gate C measured *per-contact-cown* colouring instead
(4851 behaviours/frame), which drowned in dispatch overhead and 5× manifold
rebuilds. The open question the colouring subplan must answer is the one the
user raised: **how many colours does a settled pile actually need?** If the
chromatic number is small, the available width is small regardless of how
cleanly we batch. That is the measurement Gate-1 of the colouring subplan
targets.

---

## 2. Island task-parallelism

**Idea.** Bodies that cannot possibly interact this frame (disjoint islands of
the contact graph) are solved on independent tasks. This is the coarse tier
*above* colouring. It is cheap (lock-free island assignment, O(N)) and perfectly
correct, because islands share no bodies by definition.

**Sources.**

- Jolt (`ArchitectingJoltPhysics_Rouwe_Jorrit.pdf`) builds islands lock-free and
  only invokes the colouring splitter on islands too large for one task —
  islands are the first tier, colouring the second.
- Rapier (Rust) takes the same shape: island detection feeding a Rayon
  work-stealing pool. Docs: <https://rapier.rs/docs/>.

**Map to our findings.** **Dead on arrival for our scene.** Our decisive island
probe showed a settled pile is a *single* island — the exact `LargeIslandSplitter`
trigger condition. Island parallelism and colouring are two tiers of one design;
for a dense pile only the second tier (colouring, §1) buys anything. We record
this here so the colouring subplan can state up front *why* it skips the island
tier.

---

## 3. Mass splitting (stable parallel Jacobi)

**Idea.** Solve *all* contacts in parallel (true Jacobi) in linear time, but fix
the instability that normally makes Jacobi explode on piles. Tonge's insight:
**divide each body's mass term in the *effective mass* by that body's contact
count, while still applying impulses with the full body mass.** Physically this
is equivalent to splitting each body into one virtual sub-body per contact (or
per contact *block*), with the mass divided equally among the sub-bodies; run
ordinary parallel PGS on the split system, then recombine. The paper proves this
split system converges to the same LCP solution as the unsplit one — so you get
Jacobi's parallel scaling *and* PGS's correct momentum propagation and stable
stacking, with no jitter.

**Source.**

- `Tonge-2012-MassSplittingForJitterFreeParallelRigidBodySimulation-preprint.pdf`
  — Richard Tonge, Feodor Benevolenski, Andrey Voroshilov (NVIDIA), *Mass
  Splitting for Jitter-Free Parallel Rigid Body Simulation*, ACM TOG 31(4),
  SIGGRAPH 2012. DOI <https://doi.org/10.1145/2185520.2185601>. Their GPU
  implementation runs 5 K bodies / 40 K contacts at > 60 FPS without jitter.
  A block extension (solve blocks of contacts with PGS, combine the blocks with
  Jacobi, dividing the mass term by the number of *blocks* per body) accelerates
  convergence further.

**Map to our findings — this is the correction we need.** Our ghost-row Jacobi
blew up (drift 9242, penetration 1.32) because it applied impulses computed with
the **full** effective mass on *both* sides of every seam — classic Jacobi
overshoot. Tonge says: when a body participates in *k* contacts being solved
simultaneously, scale the mass term used to *compute* each contact's impulse by
1/*k* (so the *k* parallel impulses can't collectively over-correct), but apply
the impulse with the full mass. That is the principled fix for the exact energy
injection we measured, and it preserves momentum (drift → 0). The jitter-free
Jacobi subplan adapts our ghost-row math to this recipe rather than abandoning
ghost rows.

---

## 4. Supporting / orthogonal sources

These do not change the *parallelisation* strategy but inform the *solver* whose
contacts we are parallelising. They are here for completeness and because the
teaching notes reference them.

- `smallsteps.pdf` — Macklin, Storey, Lu et al., *Small Steps in Physics
  Simulation*, SCA 2019. Shows that taking many small substeps (one solver
  iteration each) converges better per unit work than few large steps with many
  iterations. Relevant because our locked operating point is **per-substep**
  cadence, ω = 1.0 — the same "small steps" intuition.
- `XPBD.pdf` — Macklin, Müller, Chentanez, *XPBD: Position-Based Simulation of
  Compliant Constrained Dynamics*, MIG 2016. The compliant-constraint
  formulation underlying modern PBD solvers; time-step- and iteration-count
  independent stiffness.
- `PBDBodies.pdf` — Müller, Macklin et al., *Detailed Rigid Body Simulation with
  Extended Position Based Dynamics*, SCA 2020 (Vol 39, No 8). Applies XPBD to
  articulated rigid bodies with substepping; a position-level alternative to the
  velocity-impulse solver we use.

---

## 5. How the two subplans use these references

| Subplan | Primary source | Failure mode it defeats | First gate measures |
|---------|----------------|-------------------------|---------------------|
| `../PLAN-coloring.md` (islands / contact colouring) | Jolt §1, Catto §1, islands §2 | Per-contact dispatch overhead; island collapse | **Chromatic number of a settled pile** — is the available width worth it? |
| `../PLAN-jacobi.md` (jitter-free Jacobi) | Tonge §3, small-steps §4 | Pose-lag energy injection (drift 9242) | **Convergence**: mass-split Jacobi must beat the pose-lag drift and settle comparably to serial |

Both subplans follow the established discipline: a pure-Python convergence probe
first (no BOC), gated on settling quality; only if that gate passes do we map to
a BOC width probe. We implement **both** in the proven way and decide by testing.
