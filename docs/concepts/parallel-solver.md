# The parallel solver

{py:class}`~bocphysics.parallel.ParallelStepper` runs the *same* solver core as
the serial engine, but across BOC workers. It owns no physics of its own: it
reuses the engine's broad phase, solver, and bounds, and changes only **where**
each group of bodies is solved. Read {doc}`boc-in-bocphysics` first for the cown
and behavior vocabulary used below.

## Cut the world into patches

Each frame begins with the serial prologue — swept AABBs, then broad phase — and
then **partitions** the candidate pairs into patches. The default cut is
`DEFAULT_SLABS` equal-population *vertical slabs*. Vertical slabs beat a
loose-quadtree cut because they follow gravity's stacking direction: bodies
stack vertically, so a vertical cut crosses far fewer contacts, which keeps the
*seam graph* shallow (more on that below). A floor on each slab's population
collapses a small or sparse scene into fewer, fuller slabs rather than
degenerate one-body slabs.

Every patch's mutable state becomes one cown; its interior candidate pairs ride
in a second cown as a compact uid block:

```python
state_cowns.append(Cown(transport.pack_state(patch.bodies)))
intra_pairs.append(Cown(transport.pack_pairs(uid_pairs)))
```

## Interior vs. seam pairs

A candidate pair is either **interior** to one patch or a **seam** straddling
two. They are solved by two different behaviors:

- `solve_intra_substep` integrates a patch's dynamics one sub-step and resolves
  its interior pairs. It holds just that patch's state cown, so every patch's
  interior solve can run in parallel.
- `solve_boundary_substep` resolves the seam pairs that stitch two patches. It
  holds *both* patches' state cowns, so it runs only when neither neighbour is
  mid-solve.

Per-cown FIFO does all the sequencing: scheduling intra sub-step *k* before the
seams of sub-step *k* on the same cowns makes them run in that order, with no
barrier between them.

## Colour the seams to stay shallow

Each seam behavior locks two patch cowns, so the *order* seams are scheduled
sets the seam layer's critical-path depth. Scheduled arbitrarily, seams that
share a patch serialise into a chain — the dining-philosophers effect.

`colored_seam_order` avoids this by greedily **edge-colouring** the seam graph:
no two seams that share a patch get the same colour, and seams are then emitted
colour by colour. A whole colour is an independent set, so it can occupy every
patch-cown head at once, cutting the layer's depth. The colouring is a pure
function of the sorted patch-pair keys, so the schedule is deterministic and
independent of the worker count.

## Schedule a frame

`ParallelStepper.step` schedules **all** sub-steps up front and lets per-cown
FIFO order them:

```python
for _ in range(engine.num_substeps):
    for state, pairs in zip(state_cowns, intra_pairs):
        schedule_intra(state, pairs)
    for (i, j), pairs in seams:
        schedule_boundary(state_cowns[i], state_cowns[j], pairs)

schedule_writeback(state_cowns, self.engine_pinned)
```

The intra and boundary behaviors fan out across workers; the single
`schedule_writeback` lists every patch cown plus the pinned engine cown, so it
runs **last** and **on the main interpreter**, scattering each patch's final
block back onto the authoritative bodies by uid. The caller drives the frame to
completion with `quiesce` (headless) or `pump` (interactive).

## What crosses each frame

Only the small per-patch state blocks travel each step. The immutable geometry
and solve config are seeded on the noticeboard and cached per interpreter, and
geometry is re-published only when the body set changes — flagged by the
`(next_uid, count)` pair — so steady-state frames pickle nothing fat.

## A note on fidelity

The serial and parallel paths share the solver core but differ in **resolution
order**, and one known divergence is documented in the code: a seam manifold is
built *after* each patch has velocity-solved its interior, so the restitution
target it samples sees an already-damped closing speed. The two paths therefore
agree at or below the restitution threshold — the resting, stacking regime the
engine targets — but the decomposed seam suppresses restitution above it. The
gap is quantified and locked by the seam-decomposition tests rather than hidden.

See {doc}`../api/parallel` for the full reference.
