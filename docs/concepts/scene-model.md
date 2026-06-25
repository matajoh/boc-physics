# The scene model

A *scene* is a small, picklable description of the bodies that make up a
simulation. Keeping the arena as **data** rather than hard-wired construction
code lets the same engine run different worlds, lets scenes load from a JSON
file, and lets a scene cross into a worker sub-interpreter unchanged.

The model has three layers: declarative specs, a runtime emitter, and the
container {py:class}`~bocphysics.scene.Scene`.

## Specs describe bodies as data

A {py:class}`~bocphysics.scene.BodySpec` names one body — its `kind`
(`"rectangle"`, `"regular_polygon"`, or `"circle"`), colour, position, angle,
and size. It is a {py:class}`~typing.NamedTuple`, so it is immutable and trivially
serialisable. `BodySpec.build()` turns the spec into a concrete
{py:class}`~bocphysics.bodies.RigidBody`, passing `is_static` through to the
body constructor:

```python
floor = BodySpec("rectangle", "darkgreen", (0, 10), width=30, height=2)
body = floor.build(is_static=True)
```

The split between *spec* and *body* matters: the spec carries no simulation
state (no velocity, no transformed vertices), so it stays cheap to copy, store,
and round-trip through `to_dict` / `from_dict`.

## Generators emit dynamic streams

A {py:class}`~bocphysics.scene.GeneratorSpec` is the dynamic counterpart to a
`BodySpec`: where a `BodySpec` names one fixed body, a generator names a
*stream* of bodies emitted from a point at a steady `rate` (bodies per second).
The spec is immutable data; the runtime {py:class}`~bocphysics.scene.Generator`
carries the per-run mutable state — a seeded RNG and a fractional accumulator:

```python
emitter = Generator(spec)
new_bodies = emitter.update(dt)   # whole emissions this frame
```

`update(dt)` accrues `rate * dt` emissions each frame and releases the
whole-number part, carrying the remainder so the long-run rate is exact. Each
emission jitters the spawn position by `spread` and the size by `size_jitter`,
and can randomise the colour, all from the generator's own seeded RNG so a run
is reproducible.

## A Scene ties it together

{py:class}`~bocphysics.scene.Scene` is a named tuple of static specs, optional
dynamic specs, and generators, plus an optional `view_height` for the camera:

```python
scene = Scene("open_box", statics=(...,), generators=(...,))
bodies = scene.build()                  # statics first, then dynamics
emitters = scene.make_generators()      # one runtime Generator per spec
```

`build()` constructs every body (statics first), while `make_generators()`
hands back a fresh `Generator` per emitter so several independent runs can share
one immutable scene. `to_dict` / `from_dict` make the whole scene
JSON-serialisable, and `Scene.load(name)` resolves either a built-in scene name
or a path to a scene file.

## Built-in scenes

The package ships a handful of scenes — the default arena, an open-topped box
for benchmarks, and the parametric `stack`, `pyramid`, and `pachinko` scenes
whose body count scales with a row count. The parametric ones are plain
functions (`make_stack_scene`, `make_pyramid_scene`, `make_pachinko_scene`) that
return a `Scene`, so re-parametrising is just calling them with a new row count.

See {doc}`../api/scenes` for the full reference.
