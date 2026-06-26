---
name: testing-with-boc
description: "Write tests for bocpy Behavior-Oriented Concurrency code. Use when: writing pytest tests for @when behaviors, Cown scheduling, send/receive messaging, cown grouping, chained behaviors, exception propagation. Covers parameter-count rules, module-level class requirements, and the quiesce()+unwrap() result-reading pattern."
---

# Testing with Behavior-Oriented Concurrency (BOC)

This skill describes how to write tests for code that uses the `bocpy` library for
behavior-oriented concurrency. BOC schedules work as **behaviors** — decorated
functions that run once all required **cowns** (concurrently-owned data) are
available. Testing BOC programs requires specific patterns because behaviors
execute asynchronously on worker interpreters.

> **Design guidance:** if you find yourself reaching for `time.sleep`,
> `threading.Event`, polling loops, or `wait_for_*` helpers inside a
> behavior or in test code that drives one, stop and read
> `thinking-in-boc` first. The right answer is almost always to express
> the dependency through the cown graph, not through a classical
> synchronization primitive.

## Key Concepts

| Concept | Description |
|---------|-------------|
| `Cown(value)` | A concurrently-owned wrapper. Behaviors receive exclusive temporal access to the cown's `.value`. |
| `@when(*cowns)` | Decorator that schedules the function as a behavior. The decorator replaces the function with a `Cown` holding the return value. **The first N parameters bind to the N cowns; any extra values are captured as trailing parameters with defaults (`x=x`).** |
| `quiesce(timeout)` | Block until all in-flight behaviors complete **without** tearing the runtime down — further `@when` calls work immediately afterwards. Raises `TimeoutError` if quiescence is not reached. This is the synchronization point a test waits on before reading results. |
| `cown.unwrap()` | **Consume** and return a cown's value on the caller's thread, or re-raise a captured behavior exception verbatim. Call it only after `quiesce()` (or `wait()`); calling it while work is in flight raises `RuntimeError`. Consumes the cown, so a second `unwrap()` returns `None`. |
| `wait(timeout)` | Block until all scheduled behaviors complete **and tear the runtime down**. Use it once in `teardown_class` to give each test class a fresh BOC environment. |
| `notice_seed(key, value)` | Synchronously install a noticeboard entry from the primary interpreter (commits before returning). The way to seed read-mostly config before scheduling the behaviors that read it. |
| `send(tag, contents)` / `receive(tags, timeout)` | Lower-level cross-interpreter messaging. No longer the primary way to assert on behavior results — prefer `quiesce()` + `unwrap()`. Still the right tool when testing the messaging API itself (Pattern 8). |
| `TIMEOUT` | Sentinel returned as the tag by `receive` when a timeout elapses. |

### Cown count, parameter count, and captured extras

The first `N` parameters of the decorated function bind positionally to the `N`
arguments of `@when`. Any value the behavior needs from the enclosing scope
must be a **trailing parameter carrying a default**, snapshotted at schedule
time. A behavior runs in another interpreter and **cannot close over a free
variable** — doing so raises `SyntaxError` at `@when` decoration time, naming
the offending variable.

```python
# CORRECT — 1 @when arg, 1 function param
@when(x)
def good(x):
    return x.value * 2

# CORRECT — extra value captured as a trailing default by name
factor = 2
@when(x)
def with_extra(x, factor=factor):   # ``factor`` captured at schedule time
    return x.value * factor

# WRONG — closing over ``factor`` raises SyntaxError at decoration time
factor = 2
@when(x)
def bad(x):
    return x.value * factor          # free variable — rejected
```

### Use the `def _(c, x=x)` loop-capture idiom

The canonical Python idiom for snapshotting a loop variable as a default
argument is now **required** — a bare reference to the loop variable would
close over a free variable and raise `SyntaxError`:

```python
for i, c in enumerate(cowns):
    @when(c)
    def _(c, i=i):          # capture i; a bare `i` reference would fail
        send("done", i)
```

The default's name is what gets captured: `def b(c, i=i)` captures `i`, and the
rename form `def b(c, x=y)` captures `y` and binds it into param `x`. The
leading cown parameters never carry defaults. See the "Inspecting the Worker
Bindings Module" section of `.github/copilot-instructions.md` for how to inspect
the bindings module workers import.

If you want a fresh scope per iteration (e.g. to avoid sharing mutable
state between iterations), use a helper function:

```python
def _schedule(c, i):                # fresh scope per iteration
    @when(c)
    def _(c, i=i):                  # still capture i as a trailing default
        send("done", i)

for i, c in enumerate(cowns):
    _schedule(c, i)
```

### Critical rule: classes and functions must be declared at module level

Behaviors run in separate sub-interpreters. The transpiler exports the module so
workers can import it, which means **any class or function referenced inside a
`@when` behavior must be defined at module level**. A class defined inside a test
method or local function cannot be resolved by the worker and will crash.

```python
# CORRECT — class at module level
class Accumulator:
    def __init__(self):
        self.items = []
    def add(self, item):
        self.items.append(item)

def test_accumulator(self):
    acc = Cown(Accumulator())

    @when(acc)
    def _(a):
        a.value.add(42)       # Accumulator is importable ✓

# WRONG — class inside test method
def test_accumulator_bad(self):
    class Accumulator:        # local class — worker can't import it
        ...
    acc = Cown(Accumulator())

    @when(acc)
    def _(a):
        a.value.add(42)       # will crash
```

## Project Test Setup

- Tests use **pytest** and live in the `test/` directory.
- Install test dependencies: `pip install -e .[test]`
- Run: `pytest -vv`

## Pattern 1 — Read Results with `quiesce()` + `unwrap()`

Because behaviors run asynchronously, you **cannot** read a cown's result
directly in the test body right after scheduling — the behavior hasn't run yet,
and reading a cown while its producer is still in flight raises `RuntimeError`.
The modern pattern is a two-step **synchronize, then read**:

1. Schedule your behaviors with `@when`.
2. Call `quiesce(timeout)` to block until every in-flight behavior has
   completed. Unlike `wait()`, this leaves the runtime running, so the next
   test can schedule again immediately.
3. Read each result on the test thread with `cown.unwrap()`. `unwrap()`
   **consumes** the cown and returns its value (or re-raises a captured
   exception).

Wrap each suite in a class whose `teardown_class` calls `wait()` — that tears
the whole BOC runtime down once at the end of the class, giving the next class
a fresh environment.

```python
import pytest
from bocpy import Cown, quiesce, wait, when

QUIESCE_TIMEOUT = 5


def simple(x: Cown) -> Cown:
    """Schedule a behavior that doubles a cown's value."""
    @when(x)
    def do_double(x):
        return x.value * 2

    return do_double          # the returned Cown holds the result


class TestExample:
    @classmethod
    def teardown_class(cls):
        wait()                # tear the runtime down after the whole suite

    def test_double(self):
        x = Cown(3)
        result = simple(x)

        quiesce(QUIESCE_TIMEOUT)
        assert result.unwrap() == 6
```

### Why this pattern?

`@when` returns immediately — the behavior hasn't executed yet, so a plain
`result.value` read would race the worker (and `unwrap()` guards against it by
raising `RuntimeError` while work is in flight). `quiesce()` is the
synchronization barrier: once it returns, every scheduled behavior is done and
its result cown is safe to `unwrap()`. Because `quiesce()` does **not** tear the
runtime down, you can interleave `quiesce()`/`unwrap()` checkpoints with more
`@when` scheduling in the same test. The single `wait()` in `teardown_class`
resets the BOC system so behavior registries and worker caches don't leak
between test classes.

### `unwrap()` semantics

| Behavior | `unwrap()` result |
|----------|-------------------|
| Returned a value | That value. The cown is emptied to `None`. |
| Raised an exception | The exception, **re-raised verbatim** on the caller — use `with pytest.raises(...)`. |
| Returned an `Exception` object (not raised) | That object, as an ordinary value (no re-raise). |
| Already unwrapped once | `None` — `unwrap()` consumes the cown. |
| Still in flight (no `quiesce`/`wait` yet) | Raises `RuntimeError("still in flight")`. |

Consuming is also what makes move-typed values (e.g. `Matrix`) usable after the
call: the cown stops aliasing the value's single backing store, so ownership
moves to the caller's interpreter.

## Pattern 2 — Testing Nested / Chained Behaviors

Behaviors can schedule further behaviors, and a behavior chained on another's
result cown runs after it. Schedule the whole chain, `quiesce()` once, then
`unwrap()` the final result cown.

```python
def test_behavior_chain(self):
    x = Cown(2)

    @when(x)
    def step1(x):
        return x.value + 3        # 5

    @when(step1)
    def step2(s1):
        return s1.value * 4       # 20

    @when(step2)
    def step3(s2):
        return s2.value - 7       # 13

    quiesce(QUIESCE_TIMEOUT)
    assert step3.unwrap() == 13
```

To inspect intermediate steps, read more than one result cown after the same
`quiesce()`:

```python
def test_nested(self):
    x = Cown(1)

    @when(x)
    def step1(x):
        x.value *= 2              # x is now 2

        @when(x)
        def step2(x):
            x.value *= 3          # x is now 6

        return step2

    quiesce(QUIESCE_TIMEOUT)
    assert x.unwrap() == 6
```

## Pattern 3 — Multi-Cown Coordination

Pass multiple cowns to `@when` to atomically operate on several pieces of data.
The scheduler guarantees deadlock-free acquisition. Read each cown back through
its own one-cown reader behavior, then `unwrap()`.

```python
def read(c: Cown) -> Cown:
    """Schedule a behavior that returns a cown's value."""
    @when(c)
    def do_read(c):
        return c.value

    return do_read


def test_transfer(self):
    x = Cown(100)
    y = Cown(0)

    @when(x, y)
    def _(x, y):
        y.value += 50
        x.value -= 50

    x_after = read(x)
    y_after = read(y)

    quiesce(QUIESCE_TIMEOUT)
    assert x_after.unwrap() == 50
    assert y_after.unwrap() == 50
```

## Pattern 4 — Cown Grouping

When you have a dynamic number of cowns (e.g., a list), you can pass them to
`@when` as a **list** (or slice) rather than individual arguments. Inside the
behavior, that parameter is delivered as a `list[Cown]` — each element is an
acquired cown whose `.value` you can read or write.

You can mix single cowns and groups freely in any order. Each distinct argument
to `@when` becomes its own parameter in the decorated function:

| `@when(...)` arguments | Behavior parameters |
|------------------------|---------------------|
| `@when(list_of_cowns)` | `(group: list[Cown])` |
| `@when(cowns[:9], cowns[9])` | `(group: list[Cown], single: Cown)` |
| `@when(cowns[0], cowns[1:])` | `(single: Cown, group: list[Cown])` |
| `@when(cowns[:4], cowns[4], cowns[5:])` | `(g0: list[Cown], single: Cown, g1: list[Cown])` |
| `@when(cowns[0], cowns[1:9], cowns[9])` | `(s0: Cown, group: list[Cown], s1: Cown)` |

### Full group example

```python
from bocpy import Cown, when, send, receive

cowns = [Cown(i) for i in range(10)]  # values 0..9, sum = 45

# All cowns as a single group
@when(cowns)
def group_sum(group: list[Cown[int]]):
    return sum(c.value for c in group)

# Group + single cown
@when(cowns[:9], cowns[9])
def group_then_single(group: list[Cown[int]], single: Cown[int]):
    return sum(c.value for c in group) + single.value

# Single cown + group
@when(cowns[0], cowns[1:])
def single_then_group(single: Cown[int], group: list[Cown[int]]):
    return single.value + sum(c.value for c in group)

# Group + single + group
@when(cowns[:4], cowns[4], cowns[5:])
def group_single_group(g0: list[Cown[int]], single: Cown[int], g1: list[Cown[int]]):
    return sum(c.value for c in g0) + single.value + sum(c.value for c in g1)
```

### Testing grouped results

The results are all cowns, so use the same `quiesce()` + `unwrap()` pattern. You
can pass a list of result cowns as a group to a reader `@when`, or just
`unwrap()` each one after quiescing:

```python
def test_cown_grouping(self):
    expected = 45
    results = [group_sum, group_then_single, single_then_group, group_single_group]

    quiesce(QUIESCE_TIMEOUT)
    for r in results:
        assert r.unwrap() == expected
```

Mutating cowns inside a group sticks; read them back through a follow-up
behavior:

```python
def test_grouped_cown_mutation(self):
    cowns = [Cown(i) for i in range(5)]

    @when(cowns)
    def double_all(group: list[Cown[int]]):
        for c in group:
            c.value *= 2

    @when(cowns)
    def verify(group: list[Cown[int]]):
        return [c.value for c in group]

    quiesce(QUIESCE_TIMEOUT)
    assert verify.unwrap() == [i * 2 for i in range(5)]
```

### Key rules for grouping

- Pass a **list** (or slice) of cowns to `@when` — the behavior receives the
  corresponding parameter as `list[Cown]`.
- Pass a **single cown** — the parameter receives that `Cown` directly.
- You can **interleave** singles and groups in any order. The positional mapping
  between `@when(...)` arguments and the decorated function's parameters is 1:1.
- Type-annotate grouped parameters as `list[Cown[T]]` for clarity.

## Pattern 5 — Exception Propagation

If a behavior raises, the exception is captured in the result cown and the
cown's `.exception` flag is set to `True`. From the test thread, `unwrap()`
**re-raises** that exception verbatim, so assert with `pytest.raises`:

```python
def test_exception_in_behavior(self):
    x = Cown(1)

    @when(x)
    def bad(x):
        x.value /= 0              # ZeroDivisionError

    quiesce(QUIESCE_TIMEOUT)
    with pytest.raises(ZeroDivisionError):
        bad.unwrap()
```

A consumed exception is cleared, so a second `unwrap()` returns `None`. An
`Exception` object that a behavior *returns* (rather than raises) is an ordinary
value — `unwrap()` hands it back without re-raising:

```python
def test_returned_exception_is_a_value(self):
    x = Cown(1)

    @when(x)
    def returns_exc(x):
        return ValueError("not really an error")

    quiesce(QUIESCE_TIMEOUT)
    result = returns_exc.unwrap()     # returned, not raised -> no re-raise
    assert isinstance(result, ValueError)
```

> **Caveat — no bare `assert` inside a behavior.** pytest rewrites `assert`
> statements to reference a module-global `@pytest_ar` helper. The marshalled
> code object carries that reference but the worker interpreter's namespace
> lacks it, so a bare `assert` inside a `@when` body crashes with a confusing
> `NameError` on the worker. Raise an explicit exception instead
> (`raise AssertionError(...)`), or return the value and assert on it after
> `unwrap()`.

Notes:

- Writing `cown.value = ...` from inside a behavior **clears** `.exception`.
- `cown.exception` is also readable/writable inside a behavior if you want to
  inspect or manually mark error state before the result reaches the test.

## Pattern 6 — Noticeboard

The noticeboard is a global key-value store (up to 64 keys) that behaviors can
read and write **without** acquiring any cowns. Writes are non-blocking; reads
return a snapshot taken once per behavior execution.

| Function | Purpose |
|----------|---------|
| `notice_write(key, value)` | Non-blocking write from inside a behavior. |
| `notice_seed(key, value)` | **Primary-interpreter only**, synchronous write that commits before returning. Seed read-mostly config *before* scheduling the behaviors that read it. Raises `RuntimeError` if called inside a behavior. |
| `notice_update(key, fn, default=None)` | Atomic read-modify-write. `fn` and `default` must be picklable. Returning `REMOVED` deletes the entry. |
| `notice_delete(key)` | Non-blocking delete. |
| `noticeboard()` | Read-only mapping — snapshot of the noticeboard, cached for the duration of the current behavior. |
| `notice_read(key, default=None)` | Convenience: one key from the snapshot. |

From a test thread, capture a point-in-time snapshot with the `noticeboard=True`
flag on `quiesce()` — it returns a plain `dict` reflecting every write committed
by a behavior that finished before the quiesce point:

```python
snap = quiesce(QUIESCE_TIMEOUT, noticeboard=True)
assert snap.get("key") == expected
```

### Key rule: snapshot per behavior

Within a single behavior, `noticeboard()` and `notice_read()` always return
data from the **same** snapshot — even if other behaviors write in the
meantime. To see a write made by another behavior, schedule a follow-up
behavior (typically by chaining via a cown returned from `@when`), or read the
snapshot from the test with `quiesce(..., noticeboard=True)`.

```python
def test_noticeboard_roundtrip(self):
    x = Cown(0)

    @when(x)
    def step1(x):
        notice_write("greeting", "hello")

    snap = quiesce(QUIESCE_TIMEOUT, noticeboard=True)
    assert snap.get("greeting") == "hello"
```

### Atomic update

`notice_update` runs `fn(current_value)` on the scheduler and writes the
result back atomically. Lambdas and closures are **not** picklable — use a
module-level function (optionally wrapped with `functools.partial`) or an
`operator` function.

```python
from functools import partial
from operator import add

def _bump(n, by):
    return n + by

class TestCounter:
    @classmethod
    def teardown_class(cls):
        wait()

    def test_atomic_increment(self):
        x = Cown(0)

        @when(x)
        def init(x):
            notice_write("count", 0)

        @when(x, init)
        def bump(x, _):
            notice_update("count", partial(_bump, by=5))
            notice_update("count", partial(add, 3))

        snap = quiesce(QUIESCE_TIMEOUT, noticeboard=True)
        assert snap.get("count") == 8
```

### Delete via `REMOVED`

Returning the `REMOVED` sentinel from a `notice_update` callback deletes the
entry. `notice_delete(key)` is the direct form.

```python
def _drop_if_zero(n):
    return REMOVED if n == 0 else n - 1

def test_remove_via_update(self):
    x = Cown(0)

    @when(x)
    def init(x):
        notice_write("lives", 1)

    @when(x, init)
    def tick(x, _):
        notice_update("lives", _drop_if_zero)   # 1 -> 0
        notice_update("lives", _drop_if_zero)   # 0 -> REMOVED

    snap = quiesce(QUIESCE_TIMEOUT, noticeboard=True)
    assert "lives" not in snap
```

### Common noticeboard pitfalls

| Pitfall | Fix |
|---------|-----|
| Reading a value back inside the **same** behavior that wrote it | The snapshot was taken at the start of the behavior. Chain a follow-up `@when` to observe the write. |
| Passing a lambda or closure to `notice_update` | They are not picklable. Use a module-level function with `functools.partial`, or an `operator` function. |
| Asserting in the test body that `noticeboard()` contains a key | Call `noticeboard()`/`notice_read()` outside a behavior returns a never-refreshed snapshot. Read the live state with `quiesce(QUIESCE_TIMEOUT, noticeboard=True)` instead. |
| Writing more than 64 distinct keys | Excess writes are dropped with a logged warning — they do **not** raise. Keep tests within the limit (and `notice_delete` keys you no longer need). |

## Pattern 7 — Parameterized Tests

Use `@pytest.mark.parametrize` to sweep inputs. Each invocation gets its own
cowns so tests are isolated.

```python
@pytest.mark.parametrize("n", [1, 10, 15])
def test_fibonacci(self, n):
    result = fib_parallel(n)
    expected = fib_sequential(n)

    quiesce(QUIESCE_TIMEOUT)
    assert result.unwrap() == expected
```

## Pattern 8 — Testing `send`/`receive` Messaging Directly

`send`/`receive` are no longer the way to assert on behavior results (use
`quiesce()` + `unwrap()` for that). They remain the right tool when the code
under test uses the cross-interpreter messaging API itself:

```python
from bocpy import send, receive, TIMEOUT

def test_basic_messaging():
    send("tag", "payload")
    tag, value = receive("tag", 1)
    assert tag != TIMEOUT
    assert value == "payload"

def test_receive_timeout():
    tag, value = receive("tag", 0.1)
    assert tag == TIMEOUT
    assert value is None

def test_timeout_with_after_callback():
    tag, value = receive("tag", 0.1, lambda: ("fallback", 42))
    assert tag == "fallback"
    assert value == 42
```

## Pattern 9 — Complex Objects in Cowns

Mutable objects (e.g., class instances) work inside cowns. Behaviors mutate them
in-place under exclusive access.

```python
class Counter:
    def __init__(self):
        self.n = 0
    def increment(self):
        self.n += 1

def test_object_in_cown(self):
    c = Cown(Counter())

    for _ in range(10):
        @when(c)
        def _(c):
            c.value.increment()

    quiesce(QUIESCE_TIMEOUT)
    assert c.unwrap().n == 10
```

## Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| **Parameter count mismatch in `@when`** | The first N parameters must match the N `@when` cowns; any extra value must be a **trailing parameter with a default** (`x=x`). A closure over a free variable raises `SyntaxError` at decoration. |
| **Classes/functions defined inside a test** | Behaviors run in sub-interpreters that import the module. Define all classes and functions used in behaviors at **module level** so workers can resolve them. |
| Reading a result right after `@when` | The behavior hasn't run yet, and `unwrap()` raises `RuntimeError` while work is in flight. Call `quiesce(timeout)` first, then `unwrap()`. |
| `unwrap()` returning `None` unexpectedly | `unwrap()` **consumes** the cown; a second call returns `None`. Capture the value in a local on the first call if you need it twice. |
| Bare `assert` inside a `@when` body | pytest's assert rewriting references a `@pytest_ar` global the worker lacks — it crashes with `NameError`. Raise an explicit exception, or return the value and assert after `unwrap()`. |
| Forgetting `wait()` in teardown | Behavior registries and worker caches leak into the next test class. Always call `wait()` in `teardown_class`. |
| Reading `cown.value` outside a behavior | A cown must be acquired first. Read values inside `@when`, or `unwrap()` on the test thread after `quiesce()`/`wait()`. |
| Trying to capture a loop variable by closure | A behavior runs in another interpreter and cannot close over free variables (raises `SyntaxError`). Capture it as a trailing default instead: `def _(c, i=i): ...`. |
| `quiesce()` / `receive()` without a timeout | A crashed or never-firing behavior hangs the test forever. Always pass a timeout (e.g. `QUIESCE_TIMEOUT = 5`); `quiesce` raises `TimeoutError` if quiescence is not reached. |
| Non-XIData-compatible objects in cowns across interpreters | Stick to built-in types or objects that support cross-interpreter data. |
| Test function names with uppercase letters (N802) | Test names must be lowercase. E.g., `test_t_equals_transpose`, **not** `test_T_equals_transpose`, even when testing a property like `.T`. |
| Assigning `Cown(m)` to an unused variable (F841) | When the return value isn't needed (e.g., releasing a resource), use bare `Cown(m)` without assignment. |
| Using single quotes (Q000) | The project enforces `inline-quotes = double`. Use `"nan"` not `'nan'`. |
| Multi-line class docstring formatting (D205/D209) | Summary line, then blank line, then body. Closing `"""` on its own line. |
| Placing `# noqa: B023` on the `def` line instead of the violation line | `# noqa: B023` must go on the line that **references** the loop variable, not the `def _(a):` line above it. |
