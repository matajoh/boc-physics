# Chapter 2: The serial engine

[Chapter 1](01-rigid-body-physics.md) gave us a single body that falls under
gravity. A world with one body is not very interesting: it just accelerates
downward forever. The moment we add a second body, or a floor, we need to answer
two new questions every frame — *which bodies are touching?* and *what should
happen to them now that they are?*

This chapter walks the **entire single-threaded frame**, from the raw body list
to a settled, believable pile. Everything here runs on one thread; the parallel
machinery comes much later. Getting the serial engine right first is not a
detour — it is the reference the parallel version must reproduce.

The frame driver lives in
[`src/bocphysics/engine.py`](../../src/bocphysics/engine.py); the collision and
solver stages live in the neighbouring `detection.py`, `collisions.py`,
`contacts.py`, `solver.py`, and `physics.py`.

## Step first, fix afterwards

There are two broad ways to stop bodies from passing through each other.

The first is to predict the future. Given each body's position and velocity, you
can solve for the exact instant it *will* hit something, advance time to that
instant, resolve the hit, and repeat. This is **a priori** (before-the-fact)
collision detection, and it is beautifully exact:

![A priori collision detection: a ball's path to a wall is solved exactly](../images/apriori.gif)

The trouble is that the maths explodes as the world gets richer. Add gravity and
finding the moment of impact already means solving a quadratic; add a spinning,
inclined ledge and it becomes a quartic; add a hundred bodies that can all hit
each other and the approach collapses under its own algebra.

bocphysics takes the second route, **a posteriori** (after-the-fact): take a
small step *blindly*, then look for any overlaps you just created and push them
apart. The engine's own `step` says as much in its first comment:

```python
# the main problem inherent in a posteriori physics simulation
# is "tunneling" where objects pass through each other.
# to mitigate this, we subdivide the time step into smaller
# increments and resolve collisions at each step using impulses.
```

![A posteriori detection: step into an overlap, then resolve it](../images/aposteriori.gif)

The price of stepping blindly is **tunnelling**: if a body moves far enough in
one step, it can jump clean through a thin wall before anyone notices the
overlap. The defence is to chop each frame into several smaller **sub-steps**,
so no single advance is large enough to skip past an obstacle. We will see
exactly where that subdivision happens further down.

## A body is the components it has

Before tracing the frame, look at how a body joins the world. The engine does
not have separate `DynamicBody`, `StaticBody`, and `Sprite` classes. Instead a
body simply *is* the set of attributes it carries, and three **systems** decide
what it participates in:

```python
self.systems = {
    "physics": ["position", "angle",
                "linear_velocity", "angular_velocity",
                "mass", "inertia"],
    "collision": ["aabb"],
    "render": ["position", "color"]
}
```

When a body is added, `add_body` stamps a boolean for each system depending on
whether the body has all of that system's components:

```python
def add_body(self, body: RigidBody):
    for system, components in self.systems.items():
        has_system = all(hasattr(body, component) for component in components)
        setattr(body, system, has_system)
    body.uid = self.next_uid
    self.next_uid += 1
    self.bodies.append(body)
```

This is a small **entity–component–system** design, and it buys real
simplicity. A static floor is just a body that has a `position` and an `aabb`
but no `mass`, so `body.physics` is `False`: it collides and renders, but is
never integrated or pushed. There is no "is this a wall?" special-casing
anywhere in the solver — the `physics`, `collision`, and `render` flags carry
all of it.

![Modular design: the floor, ledges, and shapes share component systems](../images/modular_design.png)

You can see the flags doing their job in `step`, which solves only the bodies
that have physics and only the pairs that involve at least one of them:

```python
bodies = [body for body in self.bodies if body.physics]
pairs = [(a, b) for a, b in self.collisions if a.physics or b.physics]
```

## The frame at a glance

Here is the whole driver. It is short because each stage delegates to a
dedicated module:

```python
def step(self, dt: float):
    self.contacts.clear()
    self.update_swept_aabbs(dt)

    self.collisions.clear()
    self.broad_phase()
    bodies = [body for body in self.bodies if body.physics]
    pairs = [(a, b) for a, b in self.collisions if a.physics or b.physics]

    sub_dt = dt / self.num_substeps
    self.solve_substep(bodies, pairs, sub_dt)

    self.remove_outside()
```

Laid out as a pipeline, one frame looks like this:

```mermaid
flowchart TD
    A[clear contacts] --> B[update swept AABBs]
    B --> C[broad phase:<br/>candidate pairs]
    C --> D[keep dynamic<br/>bodies and pairs]
    D --> E{for each<br/>sub-step}
    E -->|next| F[integrate:<br/>semi-implicit Euler]
    F --> G[build manifolds:<br/>narrow phase + contacts]
    G --> H[positional correction]
    H --> I[velocity solve:<br/>N iterations]
    I --> E
    E -->|done| J[remove bodies<br/>outside bounds]
```

The **broad phase** runs once per frame and is deliberately cheap and
approximate; the **narrow phase** and the **solver** run inside the sub-step
loop, where accuracy matters. The next three sections take those stages in turn.

## Broad phase: who might be touching

With $n$ bodies there are $n(n-1)/2$ possible pairs, and running the exact
collision test on all of them is wasteful — almost none of them are anywhere
near each other. The **broad phase** is a fast, conservative filter that throws
away pairs that obviously cannot collide, leaving a short list of *candidates*
for the expensive test.

The tool for "obviously cannot collide" is the **axis-aligned bounding box**
from [Chapter 1](01-rigid-body-physics.md#the-axis-aligned-bounding-box). Two
bodies whose boxes do not overlap cannot be touching, and box overlap is four
comparisons:

```python
def disjoint(self, other: "AABB") -> bool:
    return (self.left > other.right or
            self.right < other.left or
            self.top > other.bottom or
            self.bottom < other.top)
```

Because a body moves during the frame, the engine first inflates each body's box
into a **swept AABB** that covers where it is *and* where it is heading, so a
single broad-phase pass stays valid for every sub-step. `update_swept_aabbs`
grows a dynamic body's box along `linear_velocity * dt` (statics grow only by a
small slop) and clamps the result to the world.

The simplest way to use these boxes is the brute-force scan in
`find_all_intersections_basic` — every pair, once, skipping non-colliding
bodies. It is $O(n^2)$ and perfectly correct, and it is the right thing to write
first. To quote Knuth, *premature optimization is the root of all evil*; make it
work, then make it fast.

For busier scenes the engine's default detection mode is instead a
**quadtree** (the brute-force scan stays available for comparison via
`--detect basic`). Each node owns a square region and, once it holds more than a
threshold of bodies, splits into four child quadrants. A body that fits cleanly
inside one quadrant descends into it; a body straddling several stays at the
parent. Collisions are then found by testing only bodies that share or overlap a
node, pruning whole subtrees whose box is disjoint from the body in hand:

![A quadtree over a settled pile, its cells subdividing where bodies cluster](../images/default_quadtree.png)

In the overlay above (drawn live by `--overlay quadtree`) the cells stay coarse
over empty space and subdivide where the pile is dense — exactly where the extra
resolution pays off. The structures and the two recursive walks live in
[`quadtree.py`](../../src/bocphysics/quadtree.py); the mode is selected in
[`detection.py`](../../src/bocphysics/detection.py).

Whatever the mode, the broad phase produces the same thing: a list of candidate
pairs whose boxes overlap. Some of those pairs will turn out not to be touching
at all. Deciding that is the narrow phase's job.

## Narrow phase: who is *actually* touching

The narrow phase runs the exact test on each candidate pair and, when they do
overlap, reports *how*. `detect_collision` dispatches on the shapes involved:

```python
def detect_collision(a: RigidBody, b: RigidBody) -> Collision:
    if isinstance(a, Circle):
        if isinstance(b, Circle):
            return intersect_circle_circle(a, b)
        return intersect_circle_polygon(a, b)
    elif isinstance(a, Polygon):
        if isinstance(b, Circle):
            collision = intersect_circle_polygon(b, a)
            return collision.reverse() if collision else None
        return intersect_polygon_polygon(a, b)
```

Two circles are the easy case: they touch when the distance between their
centres is less than the sum of their radii. The code compares *squared*
distances so it only pays for a square root when there really is a collision:

![Two circles collide when centre distance is under the radius sum](../images/circle_circle_diagram.png)

Polygons use the **Separating Axis Theorem** (SAT), which rests on one fact:

> If two convex shapes do not overlap, there is some axis on which their
> projections (their shadows) do not overlap either.

So the test projects both shapes onto every candidate axis — the face normals of
both polygons — and looks for a gap. The instant one axis shows a gap, the
shapes are disjoint and the function returns; since most candidate pairs are not
actually colliding, this early-out is what makes SAT cheap in practice.

![Projecting both rectangles onto each face normal; overlap on all axes means a hit](../images/rectangle_rectangle_diagram.png)

If *no* axis separates them, the shapes overlap, and the axis with the
*smallest* overlap is the most efficient direction to push them apart. That axis
is the **collision normal** and the overlap is the **penetration depth** —
together the *minimum translation vector* (MTV) that the solver will use. In
bocphysics every projection is a single batched matrix multiply rather than a
Python loop over vertices:

```python
# one batched matmul projects every vertex of both polygons onto every axis
a_proj = a.transformed_vertices @ nt
b_proj = b.transformed_vertices @ nt
...
axis = normals[depth.argmin()]   # minimum-overlap axis is the contact normal
```

A circle against a polygon is the same idea with one extra axis — the direction
from the circle's centre to the nearest polygon vertex — which catches the case
where a circle rests against a corner. Note that this is SAT, **not** GJK: there
is no Minkowski-difference simplex here, just projections onto a fixed set of
axes.

## Contact points

The normal and depth tell the solver *which way* to push, but an impulse also
needs to know *where* to push — the **contact points**. For a circle there is
exactly one, its centre plus the normal times its radius. For two polygons there
can be one (a corner resting on a face) or two (an edge lying flat on a face):

![Contact points shown as yellow dots where bodies meet](../images/contact_points.png)

`find_contact_points` finds them by sweeping every edge of one polygon against
every vertex of the other and keeping the closest. The subtlety is that
floating-point distances are never exactly equal, so "the two points that share
the minimum distance" cannot be a straight comparison. The scan keeps a running
minimum and an epsilon band around it: a distance below the band *replaces* the
manifold, a distance inside the band *adds* a second point, and a distance above
it is ignored. Each kept point also carries a `(body_uid, vertex_index)` feature
ID so the contact can be recognised again next frame.

The whole query is pure geometry with no side effects, which matters later: the
parallel workers run the identical contact code and must get byte-identical
points.

## Resolving contacts: sub-steps and projected Gauss–Seidel

Now the engine knows every overlap, its normal and depth, and its contact
points. Turning that into motion is `solve_substep`, which hands the work to the
shared solver core so the serial and parallel paths run the *same* solve:

```python
def solve_group_substep(physics, bodies, pairs, gravity, sub_dt,
                        num_substeps, num_velocity_iterations, contacts):
    for _ in range(num_substeps):
        integrate_block(bodies, gravity, sub_dt)
        resolve_manifolds(physics, pairs, num_velocity_iterations, contacts)
```

This is the heart of the a-posteriori loop, and it has **two nested levels of
iteration** that are easy to confuse:

- **Sub-steps** (`num_substeps`, default 4) chop the frame in time. Each
  sub-step integrates the bodies forward by `dt / num_substeps`, then rebuilds
  and resolves contacts. More sub-steps means smaller advances and less
  tunnelling — this is the tunnelling defence promised earlier, and for a given
  amount of work many small sub-steps also converge better than a few large ones
  ([Macklin 2019](07-references.md#macklin-2019)).
- **Velocity iterations** (`num_velocity_iterations`, default 10) happen
  *within* a sub-step. They sweep the same cached contacts over and over,
  nudging velocities until the whole coupled pile agrees.

Each sub-step does three things, in `resolve_manifolds`: **build** every pair's
manifold at the current pose (re-running the narrow phase and dropping any false
positives the broad phase let through), apply a one-shot **positional
correction** that pushes overlapping bodies apart along the MTV, then run the
**velocity solve**.

The velocity solve is **projected Gauss–Seidel** (PGS), also called sequential
impulses ([Catto 2009](07-references.md#catto-2009)). Rather than assembling and
inverting one big matrix, it resolves each
contact in turn with a small impulse, and repeats the sweep. Each contact sees
the velocity left behind by the contacts solved before it, so information
propagates through a stack — the floor pushes the bottom box, which pushes the
next, and after enough sweeps the whole tower is consistent. "Gauss–Seidel" is
exactly this *use-the-latest-value* sweep; "projected" is the clamp that keeps
each impulse physical.

The simplest mode shows the core impulse with no rotation or friction:

```python
e = self.restitution_for(contact_velocity_mag)
j = -(1 + e) * contact_velocity_mag
j /= a.inv_mass + b.inv_mass
impulse = j * normal
```

That is the textbook collision response: the impulse magnitude is

$$
j = \frac{-(1 + e)\,(\mathbf{v}_{rel} \cdot \mathbf{n})}{m_a^{-1} + m_b^{-1}},
$$

where $e$ is the **coefficient of restitution** (1 is a perfect bounce, 0 is no
bounce). Here the inverse masses from
[Chapter 1](01-rigid-body-physics.md#why-mass-and-inertia-are-stored-inverted)
pay off: a static wall has $m^{-1} = 0$, so it contributes nothing to the
denominator and receives none of the impulse — it just reflects the moving body.

### Four levels of fidelity

The `PhysicsMode` enum selects how much of the real response to model, so the
engine can double as a teaching ladder:

| Mode | What it does |
|------|--------------|
| `NONE` | Cancel the normal velocity component only — bodies stop, no bounce. |
| `BASIC` | The linear impulse above: bounce, but no spin and no friction. |
| `ROTATION` | Impulses applied at the contact points, so off-centre hits impart spin. |
| `FRICTION` | Adds a tangential impulse, bounded by a Coulomb cone, so piles rest and slide realistically. |

The two richer modes (`is_contact_mode`) cache per-contact data once per
sub-step in `prepare_collision` — the lever arms, the effective mass along the
normal, and the **restitution target**, which is sampled *once* from the
closing speed so repeated velocity iterations cannot pump energy into a resting
stack. The default mode is `FRICTION`, whose accumulated-impulse PGS clamps a
running total per contact (so a later sweep can undo an earlier over-push) and
keeps the tangential impulse inside the friction cone — stick when it can, slide
when it must. The details live in
[`physics.py`](../../src/bocphysics/physics.py).

## Where we are

We now have the whole serial frame: a cheap broad phase that nominates candidate
pairs, an exact SAT narrow phase that returns a normal and depth, a contact
generator that says where to push, and a sub-stepped projected-Gauss–Seidel
solver that turns all of it into stable, believable motion. Run it and the
`pachinko` and drop-box scenes settle into convincing piles — on one thread.

That last phrase is the catch. The per-contact Python loop at the centre of the
solver is also its bottleneck: thousands of tiny impulse calculations, each one
a handful of Python operations. [Chapter 3](03-batching.md) looks at why that
loop is slow on modern hardware, and how reshaping the data lets us replace the
loop with dense batched kernels — the groundwork the parallel solver will build
on.
