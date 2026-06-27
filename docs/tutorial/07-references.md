# Chapter 7: References

The engine in this tutorial is a teaching reconstruction, not original research.
Its serial solver, its batching, and its three parallelisation strategies all
follow well-trodden paths from games physics and real-time simulation. This page
collects the primary sources behind those choices so you can read the original
derivations and go deeper than these notes do.

The entries are grouped by the part of the engine they speak to: the
position-based solver itself and the substepping that drives it, the two families
of parallel solve (graph colouring and island tasks), the principled fix for
parallel jitter, and finally the classical impulse solver it is contrasted
against.

## The position-based solver

bocphysics resolves contacts with **Extended Position-Based Dynamics**: it
corrects positions to remove penetration, then reads velocities back from the
motion. These are the sources that solver and its substepping knob come from.

<a id="muller-2020"></a>
**Müller 2020** — Matthias Müller, Miles Macklin et al., *Detailed Rigid Body
Simulation with Extended Position Based Dynamics*, SCA 2020.
[PDF](https://matthias-research.github.io/pages/publications/PBDBodies.pdf). The
direct basis of this engine's solver: it extends XPBD to rigid bodies with the
integrate → position-solve → derive-velocity → friction/restitution sub-step that
[Chapter 2](02-serial-engine.md) implements. Read this one first.

<a id="macklin-2016"></a>
**Macklin 2016** — Miles Macklin, Matthias Müller, Nuttapong Chentanez, *XPBD:
Position-Based Simulation of Compliant Constrained Dynamics*, MIG 2016.
[PDF](http://mmacklin.com/xpbd.pdf). The compliant position-based formulation the
above builds on, and where the **compliance** parameter comes from — zero for the
rigid contacts in this engine, non-zero for soft constraints.

<a id="macklin-2019"></a>
**Macklin 2019** — Miles Macklin, Kier Storey, Michelle Lu et al., *Small Steps in
Physics Simulation*, SCA 2019. [PDF](http://mmacklin.com/smallsteps.pdf). Shows
that many small substeps beat a few large ones for the same total work — the
result behind taking a single solve per sub-step and turning up `num_substeps`
instead of iterating. The paper's headline is that a handful of substeps already
suffices; this engine measured its own knee at six to eight substeps for a
settled pile and ships a default of **eight**.

<a id="catto-2011"></a>
**Catto 2011** — Erin Catto, *Soft Constraints*, GDC 2011.
[PDF](https://box2d.org/files/ErinCatto_SoftConstraints_GDC2011.pdf). The
soft-constraint formulation that reframes a stiff constraint as a tunable
spring-damper — the same compliance idea XPBD uses, seen from the
sequential-impulse side, and the basis of the temporal Gauss–Seidel (TGS) soft
solver in Box2D v3. Useful for understanding why position-based and soft-impulse
solvers converge on the same small-substep behaviour.

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

## The classical alternative: sequential impulses

bocphysics solves contacts in position space, but the more common approach in 2D
games physics is the *sequential-impulse* solver. These two sources are the
canonical reference for it, and the origin of the graph-colouring trick
[Chapter 3](03-batching.md) uses to batch the solve.

<a id="catto-2009"></a>
**Catto 2009** — Erin Catto, *Modeling and Solving Constraints*, GDC 2009.
[PDF](https://box2d.org/files/ErinCatto_ModelingAndSolvingConstraints_GDC2009.pdf).
The canonical derivation of the sequential-impulse / projected Gauss–Seidel
solver, and the basis of Box2D. It is where the classical impulse formula and the
iterate-to-convergence loop come from — the velocity-space counterpart to the
position-space solve this engine uses.

<a id="box2d-v3"></a>
**Box2D** — Erin Catto, *Box2D: A 2D physics engine for games*.
[box2d.org](https://box2d.org/). The reference implementation of the above. Box2D
v3 adds exactly the two ideas this tutorial leans on for performance — graph
colouring of the constraint graph and wide SIMD evaluation — making it the
closest production analogue to the path from [Chapter 3](03-batching.md) onward.

---

That is the full set of sources these notes lean on. The
[concept pages](../concepts/index.rst) and the
[API reference](../api/index.rst) cover the engine as it is; everything above is
the trail to follow if you want to take it further. Thank you for reading.
