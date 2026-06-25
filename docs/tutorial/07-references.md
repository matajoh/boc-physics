# Chapter 7: References

The engine in this tutorial is a teaching reconstruction, not original research.
Its serial solver, its batching, and its three parallelisation strategies all
follow well-trodden paths from games physics and real-time simulation. This page
collects the primary sources behind those choices so you can read the original
derivations and go deeper than these notes do.

The entries are grouped by the part of the engine they speak to: the impulse
solver itself, the two families of parallel solve (graph colouring and island
tasks), the principled fix for parallel jitter, and the substepping literature
that informs the fidelity knobs.

## The impulse solver

<a id="catto-2009"></a>
**Catto 2009** — Erin Catto, *Modeling and Solving Constraints*, GDC 2009.
[PDF](https://box2d.org/files/ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf).
The canonical derivation of the sequential-impulse / projected Gauss–Seidel
solver that [Chapter 2](02-serial-engine.md) builds, and the basis of Box2D. If
you read one source here, read this one: it is where the impulse formula and the
iterate-to-convergence loop come from.

<a id="box2d-v3"></a>
**Box2D** — Erin Catto, *Box2D: A 2D physics engine for games*.
[box2d.org](https://box2d.org/). The reference implementation of the above. Box2D
v3 adds exactly the two ideas this tutorial leans on for performance — graph
colouring of the constraint graph and wide SIMD evaluation — making it the
closest production analogue to the path from [Chapter 3](03-batching.md) onward.

## Parallelising the solver: graph colouring

<a id="rouwe-2022"></a>
**Rouwe 2022** — Jorrit Rouwe, *Architecting Jolt Physics for Horizon Forbidden
West*, GDC 2022.
[Slides](https://jrouwe.nl/architectingjolt/ArchitectingJoltPhysics_Rouwe_Jorrit.pdf),
[notes](https://jrouwe.nl/architectingjolt/ArchitectingJoltPhysics_Rouwe_Jorrit_Notes.pdf).
Describes Jolt's `LargeIslandSplitter`: a greedy edge colouring of the contact
graph with a barrier between colours, so that all constraints of one colour solve
in parallel without two of them touching the same body. This is precisely the
colouring-and-barrier structure of [Chapters 3](03-batching.md) and
[4](04-converting-to-boc.md), and it in turn cites Chen et al.,
*High-Performance Physical Simulations on Next-Generation Architecture with Many
Cores*.

## Parallelising the solver: islands and tasks

<a id="jolt"></a>
**Jolt** — Jorrit Rouwe, *Jolt Physics*.
[Architecture overview](https://github.com/jrouwe/JoltPhysics/blob/master/Docs/Architecture.md).
Builds collision **islands** — connected clumps of touching bodies — lock-free
and solves independent islands in parallel, falling back to colouring only for an
oversize island. This is the patch-and-halo task parallelism
[Chapter 4](04-converting-to-boc.md) reaches for, and the reason that chapter
notes a single settled pile collapses to *one* island.

<a id="rapier"></a>
**Rapier** — *Rapier: 2D and 3D physics engines for Rust*.
[rapier.rs](https://rapier.rs/docs/). Pairs island detection with a Rayon
work-stealing thread pool — the same shape of design as bocphysics' parallel
path, with BOC's cowns-and-behaviors standing in for Rayon's tasks.

## Stable parallel Jacobi: mass splitting

<a id="tonge-2012"></a>
**Tonge 2012** — Richard Tonge, Feodor Benevolenski, Andrey Voroshilov (NVIDIA),
*Mass Splitting for Jitter-Free Parallel Rigid Body Simulation*, ACM TOG 31(4),
SIGGRAPH 2012. [DOI](https://doi.org/10.1145/2185520.2185601),
[preprint](http://www.richardtonge.com/papers/Tonge-2012-MassSplittingForJitterFreeParallelRigidBodySimulation-preprint.pdf).
The principled answer to the energy a naive parallel-Jacobi seam injects: divide
each body's effective-mass term by its contact count, then apply impulses with the
full mass. This is the literature behind [Chapter 6](06-future-work.md)'s note on
how the seam-restitution gap could be closed without giving up parallelism.

## Substepping and position-based methods

<a id="macklin-2019"></a>
**Macklin 2019** — Miles Macklin, Kier Storey, Michelle Lu et al., *Small Steps in
Physics Simulation*, SCA 2019. [PDF](http://mmacklin.com/smallsteps.pdf). Shows
that many small substeps beat a few large ones for the same total work — the
result behind [Chapter 2](02-serial-engine.md)'s default of four substeps per
frame rather than more solver iterations.

<a id="macklin-2016"></a>
**Macklin 2016** — Miles Macklin, Matthias Müller, Nuttapong Chentanez, *XPBD:
Position-Based Simulation of Compliant Constrained Dynamics*, MIG 2016.
[PDF](http://mmacklin.com/xpbd.pdf). The compliant position-based formulation that
underlies the modern small-substep solvers; useful background for why substepping
and stiffness interact the way they do.

<a id="muller-2020"></a>
**Müller 2020** — Matthias Müller, Miles Macklin et al., *Detailed Rigid Body
Simulation with Extended Position Based Dynamics*, SCA 2020.
[PDF](https://matthias-research.github.io/pages/publications/PBDBodies.pdf).
Extends XPBD to rigid bodies; a pointer to where the position-based branch of the
field goes beyond the impulse solver this engine implements.

---

That is the full set of sources these notes lean on. The
[concept pages](../concepts/index.rst) and the
[API reference](../api/index.rst) cover the engine as it is; everything above is
the trail to follow if you want to take it further. Thank you for reading.
