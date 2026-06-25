# Chapter 6: Remaining issues and directions to explore

[Chapter 5](05-experiments.md) left the engine's one real fidelity cost on the
table rather than sweeping it under it. This chapter picks that thread up and
follows it to its neighbours. None of what follows is a bug report or a promised
roadmap — the engine does what the earlier chapters claim, and its trade-offs are
measured and pinned by tests. Think of this instead as a map for the curious: if
you are looking for ways to extend this engine, these are the loose ends most
worth pulling, each with the natural next step sketched out.

## The seam-restitution divergence

This is the one limitation with a precise, documented location, so it is the
right place to start.

Recall from [Chapter 4](04-converting-to-boc.md) that the parallel solver cuts
the world into patches, solves each patch's *interior* contacts in parallel, and
then resolves the **seam** contacts that straddle two patches in a separate,
ordered pass. The trouble is one of sequencing. A seam manifold is built *after*
each patch has already velocity-solved its interior, so by the time the seam pass
samples the closing speed it needs for restitution — the "bounce" target — that
speed has already been damped by the interior solve. The seam therefore sees a
slower approach than the serial sweep would, and applies less bounce.

The effect is sharply bounded. Restitution only fires above a threshold closing
speed $v_{\text{rest}}$; below it, contacts are treated as resting and no bounce
is applied at all. The two paths therefore **agree at or below the threshold** —
the resting, stacking regime the engine is built for — and diverge only *above*
it, where a fast seam collision settles slightly softer in parallel than in
serial. That is exactly the behaviour the seam-decomposition tests in
[`test/test_parallel.py`](../../test/test_parallel.py) pin on both sides of the
threshold, so the gap cannot silently widen.

If you want to close it, the natural move is a **pre-pass**: sample every seam
contact's closing speed *before* the interior solves run, stash the restitution
targets, and have the seam pass apply them against the pre-solve speeds rather
than the damped ones. The cost is an extra read-only sweep over the seam set and
a little more state carried across the sub-step; the payoff is a parallel bounce
that matches the serial one above the threshold too. A more principled route
still is **mass splitting**, which rescales each body's effective mass by its
contact count so that a parallel Jacobi sweep across the seam stays stable
instead of injecting energy ([Tonge 2012](07-references.md#tonge-2012)). The
relevant code is `solve_boundary_substep` and `colored_seam_order` in
[`src/bocphysics/parallel.py`](../../src/bocphysics/parallel.py).

## Partitioning that does not adapt

Chapter 5 showed that the equal-population vertical-slab cut beats the quadtree
cut because it runs with the grain of gravity. But the slab cut is also
deliberately simple in ways that leave room to grow.

It is **static in shape**: the world is sliced into a fixed number of vertical
columns of equal body count every frame, with no memory of where the collision
*islands* — the connected clumps of touching bodies — actually sit. When a pile
drifts sideways or two piles merge, the slab boundaries do not follow; they are
redrawn from scratch by population alone. A cut that tracked island geometry
frame to frame, keeping a settled pile whole inside one patch instead of risking a
seam straight through it, would cut fewer load-bearing contacts and shrink the
seam pass further. Dynamic **rebalancing** — moving a boundary only when the load
on either side has drifted enough to matter — is the same idea from the
scheduling side.

The other open end is the **quadtree cut itself**. It loses in Chapter 5, and it
stays in the engine (`--quadtree-cut`) only as the honest comparison that makes
the slab cut's win measurable rather than asserted. But its loss is about
*orientation*, not about being two-dimensional: its horizontal seams slice across
vertical stacks. A 2D cut that subdivided only where it would not sever a
load-bearing column — following gravity where gravity is what matters, and
splitting freely where it is not — is an open and genuinely interesting problem.
The partition builders to start from are `build_partition` and
`build_slab_partition` in [`src/bocphysics/patches.py`](../../src/bocphysics/patches.py).

## Throughput levers left unpulled

A few performance ideas are deliberately stopped short of, each because it traded
clarity for speed in a way that did not earn its place in a teaching engine.

The interactive GUI runs a **depth-1 pipeline**: at most one physics frame is in
flight at a time, and the next is scheduled only once the previous frame's
writeback has landed (the `pump`-driven loop from
[Chapter 4](04-converting-to-boc.md), in
[`src/bocphysics/simulation.py`](../../src/bocphysics/simulation.py)). That keeps
the data flow trivial to reason about, at the cost of leaving the workers idle
during the brief window between a frame finishing and the next being dispatched.
A deeper pipeline that let frame $n+1$ begin while frame $n$'s writeback is still
draining would overlap more, at the price of having to reason about two
half-finished frames at once.

The **batched kernel** from [Chapter 3](03-batching.md) is the other lever. It is
off by default and, more to the point, it runs only on the *serial* path today.
It changes how a group of contacts is evaluated; partitioning changes *where*
they are evaluated; and the two are orthogonal. Composing them — running the
colour-batched matrix kernels *inside* each parallel patch — is the obvious
multiplier the engine sets up but does not yet take. The kernels live in
[`src/bocphysics/kernel.py`](../../src/bocphysics/kernel.py) and the A/B toggle in
[`src/bocphysics/solver.py`](../../src/bocphysics/solver.py).

Finally, there is no **sleeping**. A pile that has fully settled is re-solved in
full every frame, even though nothing is moving. Most production engines let an
island whose energy has stayed near zero for a while go to sleep, skipping it
until something touches it. Adding that would mean tracking per-island rest state
and waking neighbours on contact — a clean fit for the cown-per-patch structure,
since a sleeping patch is simply one whose behavior schedules no work.

## Things to be candid about

Not everything here is a feature waiting to be built; some of it is just worth
stating plainly so the numbers are not over-read.

The parallel path carries a real **fixed overhead** — cutting the world, packing
state blocks, scheduling behaviors — that only pays off once the scene is dense
enough to keep the workers busy. Chapter 5's early frames are *slower* in
parallel for exactly this reason. There is a crossover point, and below it the
serial path is simply the right tool.

The speed-up is also **worker-count sensitive** and machine-dependent. Eight
workers is a reasonable default on the hardware the numbers were captured on, but
the right count depends on core count, scene density, and how the slabs happen to
balance; there is no single best value, which is why the worker count is a flag
rather than a constant.

And the parallel path is, by construction, **not bit-identical** to the serial
one. Resolving a pile's contacts in a different order than a single sweep would
can never reproduce that sweep exactly, so the batched and parallel paths are
validated against *settling-band* tests — does the pile come to rest in the right
place, within tolerance? — rather than the bit-exact golden master that guards the
serial scalar solver. That is the correct contract for a re-ordering, but it is a
weaker one, and worth knowing when you reach for a regression test.

## Where to go from here

Every direction above is a place where this engine chose clarity over the last
increment of speed or fidelity, and said so. That is the right call for code that
doubles as lecture notes — but it also means the interesting extensions are
unusually well sign-posted. Pick the seam pre-pass for a self-contained solver
exercise, the island-aware cut for a meatier scheduling project, or the
batched-inside-patches composition for a pure performance win.

[Chapter 7](07-references.md) collects the papers and talks the solver and its
parallelisation are built on, so you can follow any of these threads back to the
source material.
