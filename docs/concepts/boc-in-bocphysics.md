# Behavior-Oriented Concurrency in bocphysics

bocphysics parallelises its physics step with
[Behavior-Oriented Concurrency (BOC)](https://pypi.org/project/bocpy/), not
threads and locks. BOC eliminates data races and deadlocks *by construction*:
mutable data lives inside **cowns** (concurrently-owned wrappers), and work runs
as **behaviors** that the scheduler dispatches only once every cown a behavior
needs is free. This page explains the handful of BOC ideas the engine relies on;
{doc}`parallel-solver` then shows how they compose into a parallel frame.

## Cowns hold the mutable state

A {py:class}`Cown` wraps a piece of data that may be touched concurrently. Code
may read or write `cown.value` **only** inside a behavior that holds that cown.
In the parallel stepper, each patch of the world is one such cown: its mutable
state is packed into a single dense matrix block and dropped into a `Cown`.

```python
state_cown = Cown(transport.pack_state(patch.bodies))
```

Because the block is the only mutable thing crossing between frames, the data
that travels each step stays small.

## Behaviors run when their cowns are free

A **behavior** is a function scheduled with `@when(*cowns)`. The scheduler runs
it on some worker sub-interpreter once it can acquire exclusive access to every
listed cown. The first N parameters bind to those N cowns:

```python
@when(state_cown, pairs_cown)
def _intra(state, pairs):
    solve_intra_substep(state, pairs.value)
```

Two rules of the model show up throughout the engine:

- **A behavior runs in another interpreter.** Anything it needs beyond its cowns
  must be a *trailing parameter with a default* — a behavior cannot close over a
  free variable. Referencing an enclosing name that is not a parameter is
  rejected at decoration time.
- **Module-level definitions only.** The functions and classes a behavior uses
  must be importable, because each worker imports the module fresh.

## Per-cown FIFO is the only ordering

The single synchronisation primitive the stepper uses is **per-cown FIFO
order**: behaviors scheduled on the same cown run in schedule order. There are no
barriers, events, or locks. That one guarantee is enough to sequence the
sub-steps of a patch — schedule sub-step 1 then sub-step 2 on the same state
cown, and they run in that order, on whichever worker is free.

A behavior that must run *after* several others simply lists all of their cowns.
The final writeback lists *every* patch cown, so per-cown FIFO places it dead
last automatically — it cannot start until every prior behavior on every patch
has finished.

## The noticeboard carries set-once data

Not everything belongs in a cown. Immutable, read-mostly data — the body
geometry and the solve configuration — rides the **noticeboard**, a key/value
store the runtime caches per interpreter:

```python
notice_seed(CONFIG_KEY, config)        # seed once on the main interpreter
config = notice_read(CONFIG_KEY)        # cached read inside a worker
```

Seeding happens once; each worker reads and caches the value on first use, so
the fat geometry is pickled to a worker only when the body set actually changes,
not every frame. Keeping geometry off the per-frame path is what makes the small
state block the *only* thing that crosses each step.

## Pinning work to the main interpreter

A {py:class}`PinnedCown` forces any behavior that holds it onto the main
interpreter. The stepper pins the engine so the final writeback — which mutates
the authoritative {py:class}`~bocphysics.bodies.RigidBody` objects — runs back on
main, where those objects live.

With these pieces — cowns for mutable patch state, behaviors ordered by per-cown
FIFO, the noticeboard for set-once data, and one pinned writeback — the parallel
solver needs no explicit synchronisation at all. See {doc}`parallel-solver`.
