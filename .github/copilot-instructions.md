# bocphysics — Copilot Instructions

## What bocphysics Is

`bocphysics` is a 2D rigid-body physics engine written in Python on top of the
[`bocpy`](https://pypi.org/project/bocpy/) Behavior-Oriented Concurrency
library. It is used both as a runnable simulation (entry point
`simulation = bocphysics:main`) and as a teaching aid for the Cambridge 4M26
Tripos: source files double as lecture notes and a revision aid, so
readability is a first-class concern alongside correctness.

This project depends on [`bocpy`](https://pypi.org/project/bocpy/), a Python
library implementing **Behavior-Oriented Concurrency (BOC)**. BOC eliminates
data races and deadlocks by construction: data lives inside **cowns**
(concurrently-owned wrappers), and code runs as **behaviors** that the
scheduler dispatches once all required cowns are available. Workers run in
sub-interpreters and are truly parallel on Python 3.12+.

Before writing or reviewing concurrency code in this project, read the
`thinking-in-boc` skill at `.github/skills/thinking-in-boc/SKILL.md`. It is
short, opinionated, and prevents the most common class of mistake: reaching
for threads-and-locks primitives instead of expressing the dependency through
the cown graph.

## BOC programming primer

The full reference is the `thinking-in-boc` skill. The short version:

| You wrote | What you almost certainly meant |
|-----------|---------------------------------|
| `time.sleep(...)` in a polling loop | Schedule a behavior on the cown the predicate depends on. |
| `while not <flag>: ...` busy-wait | Make `<flag>` a cown and `@when(flag)` a behavior on it. |
| `threading.Event` / `Condition` / `Lock` | A cown plus a behavior chain. |
| `Future` / `Queue.get()` to ferry a value out | `return` the value from a behavior; `@when(behavior)` reads it. |
| Loop inside one behavior to process many items | A **behavior loop**: process one chunk, then `@when(state)` again. |
| `wait_for_*` helpers / polling | Replace with `@when(downstream_cowns)`; let the cown graph do the ordering. |

The five replacement patterns you should know cold:

1. **Sequencing on data** — `@when(x)` runs after any prior `@when(x)`
   completes. That is the entire ordering mechanism.
2. **Multi-cown / barrier** — `@when(a, b, c)` when you know the cowns at
   write-time; `@when(cowns)` (a single list arg) when the set is dynamic.
3. **Happens-after across unrelated data** — chain on the prior behavior's
   result cown: `@when(x, prior_behavior)`.
4. **Run when any worker is free** — `@when()` (no args) for fire-and-forget
   follow-ups that should not block the current behavior.
5. **Behavior loops** — to process work in chunks, schedule the next iteration
   from inside the current one with `@when(state)`. Never write a `while`
   loop inside a single behavior.

### Public bocpy API (the only surface you should touch)

| Symbol | Purpose |
|--------|---------|
| `Cown[T]` | Typed wrapper for concurrently-owned data. Read/write `.value` only inside an `@when` that holds the cown. `.exception` is `True` if the behavior that produced this cown's value raised. |
| `@when(*cowns)` | Decorator. Schedules the function as a behavior with exclusive access to the listed cowns. The decorated function must take **exactly** as many parameters as `@when` got arguments. Default args count — do not use them. |
| `send(tag, contents)` | Cross-interpreter message send. Non-blocking. |
| `receive(tags, timeout, after)` | Selective receive. Returns `(TIMEOUT, None)` on timeout. |
| `drain(tags)` | Clear all queued messages for the given tag(s). |
| `set_tags(tags)` | Pre-assign tags to queues; clears all messages. |
| `TIMEOUT` | Sentinel returned by `receive` on timeout. |
| `noticeboard()` | Read a per-behavior snapshot of the global key-value store. |
| `notice_read(key, default)` | Read a single key from the snapshot. |
| `notice_write(key, value)` | Non-blocking write. |
| `notice_update(key, fn, default)` | Atomic read-modify-write. `fn` must be picklable (module-level function or `functools.partial`). Return `REMOVED` to delete. |
| `notice_delete(key)` | Non-blocking delete. |
| `notice_sync()` | Flush this thread's pending noticeboard writes before releasing the current cown. Use when a downstream behavior must observe your write. |
| `REMOVED` | Sentinel for deleting via `notice_update`. |
| `wait(timeout)` | Block until all scheduled behaviors complete; stops the runtime. |
| `start(workers, export_dir, module)` | Manually start the runtime (auto-called on first `@when`). |

### Hard rules

- **Classes and functions used inside a behavior must be defined at module
  level.** Behaviors run in sub-interpreters that import your module; locally-
  defined classes inside a test method or function cannot be resolved.
- **Parameter count must match `@when` argument count exactly.** A mismatch
  crashes the worker silently and the behavior never completes — your test
  will hang unless `receive` has a timeout.
- **Do not use `def _(c, x=x)` to snapshot a loop variable.** The transpiler
  already snapshots captures by value at schedule time. Adding `x=x`
  introduces an extra parameter that breaks the call.
- **No `time.sleep`, `threading.*`, atomics, or polling inside a behavior or
  in code that drives one.** Those primitives are only correct (a) outside
  the runtime when talking to it (e.g. a test thread blocking on `receive`
  for an assertion), (b) inside `wait()` itself, or (c) inside bocpy's own
  C internals. If you are not in one of those three places, re-derive the
  design through the cown graph.

## Project layout

- `src/bocphysics/` — the package source. Notable modules:
  - `__init__.py` — defines `main()` and the `simulation` console-script entry
    point; argument parsing for `--mode`, `--detect`, `--size`, etc.
  - `simulation.py` — top-level `Simulation` class; the pygame loop and the
    glue between rendering and the physics engine.
  - `engine.py` — `PhysicsMode` enum and the physics step driver.
  - `physics.py` — integration and force application.
  - `bodies.py` — rigid-body representations (convex polygons).
  - `collisions.py` — collision response (impulse / friction).
  - `contacts.py` — contact-point generation.
  - `detection.py` — broad/narrow phase entry points.
  - `quadtree.py` — spatial index for broad-phase detection.
  - `config.py` — `Resolution`, `DetectionKind`, tunable constants.
- `pyproject.toml` — `setuptools` build, `bocpy` and `pygame` runtime deps,
  `pytest` as an optional `[test]` extra, `simulation` console script.
- `src/bocphysics.egg-info/` — generated by editable installs; do not edit.
- Tests live under `test/` (create the directory the first time you add a
  test; follow the patterns in `testing-with-boc`).

C extensions: if any are added later, they live alongside the package source
and are picked up by `pip install -e .` — that triggers a rebuild against the
active interpreter's headers, so always reinstall after switching venvs.

## Scratch and temporary files

Use the `.copilot/` directory at the repo root for **all** temporary files:
diffs saved for review, scratch scripts, generated transpiler output, ad-hoc
notes, intermediate command output, plan/review artifacts. The directory is
gitignored.

- **Do not use `/tmp`** or any other system temp location. Keeping scratch
  files inside the repo means they survive across tool calls in the same
  session and are easy to find again.
- **Look in `.copilot/` first** when searching for prior scratch artifacts.
  Standard search tools respect `.gitignore`, so you may need to pass
  include-ignored flags to see these files.
- Create the directory with `mkdir -p .copilot` if it does not yet exist.

## Build and test

Always activate the project virtual environment first. This project uses:

- `.env314` — the default (Python 3.14) venv used for development and tests.

If the user does not specify a venv at the start of a session, suggest the
default and wait for confirmation. Never run `pip`, `pytest`, `python`, or
any project command outside the activated venv.

Typical workflow:

```bash
source .env314/bin/activate
pip install -e .[test]       # editable install with test deps
pytest                       # run the test suite
```

Re-installing in a fresh venv triggers a rebuild of any C extensions against
that interpreter's headers.

---

## How to work on this project

### Move slow to go fast

Break every task into small, testable steps. Do not attempt to fix or implement
multiple things in a single pass. Each step should be independently verifiable
before moving on.

### Plan before you act

Before making changes:

1. **Write a plan** — outline the steps you intend to take. Save it in a
   session memory plan file so it stays in context throughout the task.
2. **Get the plan approved** — present the plan and wait for explicit approval
   before writing any code.
3. **Update the plan as you go** — record progress, findings, and any
   deviations in the plan file so context is never lost.

### Baseline the tests

Before modifying any code:

1. Run the full test suite and record the results in your plan file.
2. Note any pre-existing failures so you can distinguish them from regressions
   you introduce.
3. Keep this baseline in context for the duration of the task.

### Review every non-trivial change

After implementing a change, run the **review-loop** skill to get an
independent review. This may be skipped for trivial changesets, but you do
not decide what qualifies as trivial — ask for approval first.

For a complete pre-merge audit, use **branch-review** instead, which runs
three constructive reviewer lenses plus an adversarial gap-analysis pass over
the branch diff.

For non-trivial design work (multi-subsystem changes, architecture
decisions), use **multi-perspective-plan** to draft and stress-test the plan
with competing lens subagents before any code is written.

Other skills available in `.github/skills/`:

- **thinking-in-boc** — the BOC mental model. Read this any time you catch
  yourself reaching for a classical synchronization primitive.
- **testing-with-boc** — how to write pytest tests against `@when`, `Cown`,
  noticeboard, exception propagation, and cown grouping, including the
  `send`/`receive` assertion pattern.
- **c-extensions-with-bocpy** *(only if this project has C extensions)* —
  how to write a native type whose instances can live inside a `Cown` and
  cross worker sub-interpreters via the bocpy public C ABI. Covers
  `XIDATA_REGISTERCLASS`, multi-phase init, the producer/consumer callback
  pair, and the proto-Region ownership discipline. Read this **before**
  designing any C type that will be wrapped in a `Cown`.
- **commenting-c-and-python** *(only if this project has C extensions)* —
  the doc-comment conventions used across bocpy's own C and Python sources.

The user is your collaborator. If you are unsure how to address a reviewer's
comment, ask rather than guessing.

### Fix root causes, not symptoms

When diagnosing a bug or unexpected behavior, trace the problem to its
origin. Do not apply surface-level patches that silence an error or make a
test pass without understanding **why** the failure occurred. A fix that
papers over a symptom often hides a deeper defect that will resurface later
in a harder-to-debug form.

Before writing a fix:

1. **Reproduce** — confirm the failure with a minimal case.
2. **Trace** — follow the control and data flow back to where things first go
   wrong, not where they are first observed.
3. **Understand** — articulate the root cause in the plan file before
   proposing a change.
4. **Verify** — ensure the fix addresses the root cause and does not merely
   suppress the symptom.

If the root cause is ambiguous or spans multiple subsystems, surface that
uncertainty rather than guessing. Ask for guidance.

For concurrency bugs specifically: if you find yourself reaching for a
classical synchronization primitive to "fix" the issue, stop and re-read
`thinking-in-boc`. The root cause is almost always a missing cown
dependency, not a missing lock.

### You do not commit code

All git operations (commit, push, branch management) are the user's
responsibility. Do not run `git commit` or `git push`.

### Test your changes

Every change must be tested:

- If test coverage already exists, run the relevant tests and confirm they
  pass.
- If coverage does not exist, **add tests** before considering the change
  done. Follow the patterns in `testing-with-boc`.
- Where appropriate, make tests **fuzzable** — parameterize over random or
  generated inputs so they surface bugs that hand-picked cases might miss.
