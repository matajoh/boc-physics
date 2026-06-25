# The engine step

{py:class}`~bocphysics.engine.PhysicsEngine` owns the list of rigid bodies and
advances them one frame at a time. A frame is an *a posteriori* step: bodies are
moved first, then any overlaps are detected and resolved with impulses. The
whole frame is `PhysicsEngine.step(dt)`.

## Components decide a body's role

When a body is added, `add_body` tags it with three boolean flags by checking
which attributes it carries:

| Flag        | Present when the body has…                                  |
|-------------|-------------------------------------------------------------|
| `physics`   | position, velocity, mass, inertia (a dynamic, moving body)  |
| `collision` | an AABB (it participates in detection)                      |
| `render`    | a position and colour (it can be drawn)                    |

A static body lacks velocity and mass, so `physics` is `False`: statics never
move. This is why a static can be drawn (`render`) yet skipped by the integrator
and by out-of-bounds culling. Each body is also stamped with a monotonic `uid`,
a stable identity the parallel path uses to scatter results back by body rather
than by list position.

## Anatomy of a frame

`step(dt)` runs the same prologue every frame, then hands the whole world to the
solver as one group:

1. **Swept AABBs** — `update_swept_aabbs` grows each body's tight box along its
   frame motion. One broad-phase pass over these padded boxes then yields
   candidate pairs valid for *every* sub-step, so detection runs once per frame
   rather than once per sub-step.
2. **Broad phase** — `broad_phase` finds candidate overlapping pairs using the
   chosen {doc}`detection <../api/detection>` strategy (a simple sweep or a
   quadtree).
3. **Group selection** — the frame solves every dynamic body together and every
   candidate pair that has at least one dynamic endpoint. Static–static pairs
   are dropped (two immovable bodies can never resolve), and statics are kept
   out of the integrated set since they never move.
4. **Solve** — that one group is advanced over the frame's sub-steps.

The engine deliberately does *not* split the world into independent contact
islands. For the Gauss–Seidel solver, resolving disjoint groups separately is
identical to resolving them together — same result, same work — so the
partition would buy nothing without sleeping or per-island parallelism, neither
of which the serial engine does. The parallel stepper *does* partition, but into
gravity-aligned slabs rather than islands; see {doc}`parallel-solver`.

## Sub-stepping beats tunnelling

The core hazard of an *a posteriori* method is **tunnelling**: a fast body can
pass clean through a thin one within a single step. The engine mitigates this by
subdividing `dt` into `num_substeps` smaller increments and resolving contacts
at each one.

The default substep solver (`solve_substep`) separates two costs that the
naive solver conflates:

- Each **sub-step** integrates the bodies once, then builds every pair's contact
  manifold once — the geometry barely moves within a sub-step, so the manifold
  is reused.
- The **velocity solver** then iterates `num_velocity_iterations` times over
  those cached manifolds, converging the coupled contacts *without* paying the
  narrow-phase cost again.

This work is delegated to the shared {doc}`solver core <../api/engine>` so the
serial engine and the parallel workers run the identical solve.

## Two solver back-ends

The velocity solve has two interchangeable implementations behind the same
interface, selected by the `use_batched_solver` flag in
{py:mod}`bocphysics.solver`:

- **Scalar PGS (default)** — projected Gauss–Seidel as a plain Python loop:
  prepare each contact once, then sweep the constraints `num_velocity_iterations`
  times, applying impulses one contact at a time so each read sees the previous
  update. In `FRICTION` mode the sweep is ordered apex-first (top-down, along
  gravity) with accumulated impulses, which settles tall stacks faster.
- **Batched solver** — the same projected Gauss–Seidel, but vectorised. It
  **graph-colours** the contacts so that no two constraints in a colour share a
  body, then resolves a whole colour at once as dense matrix kernels rather than
  a per-contact Python loop. A colour is an independent set, so applying it in
  one batched step matches the sequential sweep while amortising the interpreter
  overhead across many contacts. This is the same colouring idea the parallel
  seam scheduler uses, applied here within a single solve.

Both back-ends converge to the same contact behaviour; the batched path trades a
colouring pass for far fewer Python-level operations, which pays off as the
contact count grows.

## After the solve

Finally, `remove_outside` culls dynamic bodies whose swept box has drifted
entirely outside the simulation bounds. Statics are deliberately excluded — they
never move, so culling one would only desynchronise the cached static render
layer.

The same prologue (swept AABBs, broad phase, candidate pairs) is what the
parallel stepper reuses; it changes only *where* the per-group solve runs. See
{doc}`parallel-solver` for that story.
