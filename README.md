# bocphysics

A 2D rigid-body physics engine written in Python on top of
[`bocpy`](https://pypi.org/project/bocpy/), a library for **Behavior-Oriented
Concurrency (BOC)**. The project doubles as a teaching aid for learning to
program with BOC: the source is written to be read, so readability sits alongside
correctness as a first-class concern.

## What it does

- Convex-polygon and circle rigid bodies with mass, inertia, and friction.
- Position-based (XPBD) collision handling: each frame runs several sub-steps,
  and each sub-step integrates, re-detects (limiting tunnelling), projects the
  bodies out of penetration, then applies restitution and friction. Stable
  stacks come from the sub-step count (default 8).
- Broad-phase detection via a quadtree spatial index (or a brute-force scan).
- An opt-in BOC worker solver (`--parallel`) that cuts the world into
  equal-population vertical slabs and fans each slab's solve across workers.
- Declarative, picklable scene specifications (`bocphysics.scene`).
- An interactive pyglet front-end and a headless benchmark.

## Install

bocphysics requires **Python 3.12 or newer** (the parallel solver uses
per-worker sub-interpreters, which are only truly parallel on 3.12+).

Install the released package from PyPI:

```bash
pip install bocphysics
```

Or install from a clone of this repository — add the `test` extra to run the
suite:

```bash
git clone https://github.com/matajoh/boc-physics.git
cd boc-physics
pip install -e .[test]
```

The interactive front-end opens a window via [pyglet](https://pyglet.org), so
it needs a display. The headless [benchmark](#benchmark) runs without one.

## Run the simulation

The install adds a `simulation` console script:

```bash
simulation                       # the default interactive arena
simulation --scene open_box      # an open box to click shapes into
simulation --parallel --debug    # the default arena on BOC workers, with the debug overlay
```

### Controls

| Input | Action |
|-------|--------|
| Left-click | Spawn a circle at the cursor |
| Right-click | Spawn a polygon at the cursor |
| Space | Pause / resume the simulation |
| Escape | Close the window |

## Scenes

Pick a built-in scene with `--scene`, or pass a path to a scene JSON file
(see [`bocphysics.scene`](src/bocphysics/scene.py)). The built-ins are:

| Scene | What it is |
|-------|------------|
| `default` | A floor and two angled ledges — the interactive sandbox. |
| `open_box` | An open-topped box to click or stream shapes into. |
| `stack` | A settling vertical column of boxes. |
| `pyramid` | The torque-prone brick pyramid (a stability stress test). |
| `golden` | The deterministic seeded scatter the golden-master test settles. |

The `stack` and `pyramid` scenes are parametric: `--levels N` sets their row
count, e.g. `simulation --scene pyramid --levels 6`.

## Command-line options

| Flag | Values (default) | Purpose |
|------|------------------|---------|
| `--scene` | name or JSON path (`default`) | Choose the arena. |
| `--levels` | int (auto) | Row count for `stack` / `pyramid`. |
| `--detect` | `quadtree`, `basic` (`quadtree`) | Broad-phase algorithm. |
| `--parallel` | flag | Solve each frame across BOC worker sub-interpreters. |
| `--workers` | int (auto) | Worker count when `--parallel` is set. |
| `--size`, `-s` | `WxH` (`1200x900`) | Window size. |
| `--show-contacts` | flag | Draw contact points. |
| `--debug`, `-d` | flag | Overlay debug information. |

Add `--parallel` to any scene to watch the BOC worker solver, which cuts the
world into equal-population vertical slabs and fans each slab's solve across
workers.

## The per-frame step

Each frame builds the broad phase once, then solves all dynamic bodies and
their candidate pairs together as a single group with a position-based (XPBD)
solver. The work is split into several **sub-steps** (default 8); convergence
comes from the sub-step count rather than an inner iteration loop. Each sub-step
integrates the dynamic bodies, builds every pair's contact constraint (the
narrow phase), projects the bodies out of penetration (the position solve),
derives velocities from the position change, then applies restitution and
friction (the velocity solve). Out-of-bounds bodies are pruned at the end of the
frame.

```mermaid
flowchart LR
    A[broad phase: quadtree pairs] --> C[sub-step: integrate bodies]
    C --> D[narrow phase: build contact constraints]
    D --> E[position solve: project out of penetration]
    E --> G[velocity solve: restitution and friction]
    G -->|next sub-step| C
    C -->|frame done| F[remove out-of-bounds]
```

## Benchmark

[`bench/drop_box.py`](bench/drop_box.py) is a headless perf and convergence
probe. It **streams** a mix of circles and polygons into an open box over the
course of the run, steps the engine without a window, and reports wall-clock
cost per frame plus two convergence proxies: total **kinetic energy** (should
decay toward rest) and total **penetration depth** (should stay bounded). Spawn
placement is drawn from a seeded Matrix PRNG (`--seed`, default 0): each of the
five runs uses a different but deterministic seed, so the runs genuinely differ
yet the whole sweep reproduces exactly. Timing still depends on the machine and
load, so treat the tables below as a trend, not a contract — they were captured
on an Intel Core i7-14700F (28 logical cores) with turbo boost disabled for
stable timing, under Linux, on CPython 3.14.4 with bocpy 0.13.0.

Streaming the drops (rather than releasing one clump) takes the scene through
distinct stages — scattered singletons, then several separate piles, then one
merged pile — which is what exercises the collision **islands** the engine
resolves independently.

```bash
python bench/drop_box.py --shapes 80 --frames 300
python bench/drop_box.py --shapes 80 --frames 300 --batched
python bench/drop_box.py --shapes 80 --frames 300 --snapshot 40,150,300
python bench/drop_box.py --shapes 80 --frames 300 --video drop_box.mp4
python bench/drop_box.py --shapes 80 --frames 300 --parallel --workers 8
```

Add `--parallel` to solve each frame across BOC workers. The parallel run cuts
the world into equal-population vertical slabs by default; pass `--quadtree-cut`
to benchmark the loose-quadtree fallback or `--slabs N` to set the slab count.
Add `--batched` (serial or parallel) to swap the per-contact Python loop for the
colour-batched velocity kernel.

### Serial scalar (80 shapes, 300 frames, quadtree)

Averaged over five runs, reported as mean ± one standard deviation.

| Frame | ms/frame | Kinetic energy | Penetration |
|------:|---------:|---------------:|------------:|
|    30 |   0.18 ± 0.05 |    129.73 ± 7.82 |  2.0000 ± 0.0000 |
|    60 |   0.43 ± 0.10 |    938.91 ± 35.91 |  2.0000 ± 0.0000 |
|    90 |   0.75 ± 0.14 |   3065.92 ± 140.69 |  2.0000 ± 0.0000 |
|   120 |   1.07 ± 0.15 |   6300.02 ± 766.00 |  2.0000 ± 0.0000 |
|   150 |   1.50 ± 0.13 |   6489.65 ± 293.70 |  2.0000 ± 0.0000 |
|   180 |   2.37 ± 0.09 |   5861.46 ± 779.70 |  2.0004 ± 0.0005 |
|   210 |   4.05 ± 0.35 |   5473.43 ± 402.60 |  2.0160 ± 0.0186 |
|   240 |   6.38 ± 0.41 |   5170.93 ± 884.35 |  2.0183 ± 0.0115 |
|   270 |   8.79 ± 0.41 |   4124.11 ± 398.76 |  2.0575 ± 0.0523 |
|   300 |  11.82 ± 0.73 |   1729.56 ± 104.55 |  2.0332 ± 0.0142 |

Mean 3.73 ± 0.07 ms/frame over the five runs, with the XPBD solver running 8
sub-steps per frame. Cost climbs as bodies accumulate and islands merge; kinetic
energy peaks mid-run while shapes are still falling, then collapses as the pile
settles. Penetration holds near 2 throughout — the convergence the 8 sub-steps
buy us.

### Serial batched (80 shapes, 300 frames, quadtree)

The same scene with `--batched`, which swaps the per-contact Python loop for the
colour-batched velocity kernel. Same XPBD physics, same seeded sweep.

| Frame | ms/frame | Kinetic energy | Penetration |
|------:|---------:|---------------:|------------:|
|    30 |   0.16 ± 0.03 |    129.73 ± 7.82 |  2.0000 ± 0.0000 |
|    60 |   0.44 ± 0.13 |    938.91 ± 35.91 |  2.0000 ± 0.0000 |
|    90 |   0.78 ± 0.18 |   3065.92 ± 140.69 |  2.0000 ± 0.0000 |
|   120 |   1.04 ± 0.08 |   6300.02 ± 766.00 |  2.0000 ± 0.0000 |
|   150 |   1.51 ± 0.11 |   6489.65 ± 293.70 |  2.0000 ± 0.0000 |
|   180 |   2.53 ± 0.15 |   5862.40 ± 778.36 |  2.0004 ± 0.0005 |
|   210 |   4.29 ± 0.34 |   5478.71 ± 381.76 |  2.0320 ± 0.0291 |
|   240 |   6.85 ± 0.19 |   4976.59 ± 744.52 |  2.0442 ± 0.0315 |
|   270 |   9.28 ± 0.54 |   4064.36 ± 295.75 |  2.0241 ± 0.0180 |
|   300 |  12.82 ± 0.83 |   1701.11 ± 168.97 |  2.0519 ± 0.0303 |

Mean 3.97 ± 0.08 ms/frame — within noise of the scalar serial sweep, marginally
slower here. At this body count the colour-batching overhead roughly cancels the
saving from dropping the per-contact loop, so it neither helps nor hurts; the
kernel earns its keep at higher contact counts. Penetration and kinetic energy
track the scalar run, as the same XPBD physics demands.

### BOC parallel (slab cut, 8 workers)

The same scene under `--parallel --workers 8`, which cuts the world into
equal-population vertical slabs and fans each slab's solve across BOC workers.
Averaged over five runs, mean ± one standard deviation (the same seeded sweep as
the serial runs).

| Frame | ms/frame | Kinetic energy | Penetration |
|------:|---------:|---------------:|------------:|
|    30 |   0.73 ± 0.02 |    129.73 ± 7.82 |  2.0000 ± 0.0000 |
|    60 |   1.51 ± 0.39 |    939.05 ± 35.97 |  2.0000 ± 0.0000 |
|    90 |   2.09 ± 0.30 |   3066.06 ± 140.84 |  2.0000 ± 0.0000 |
|   120 |   2.38 ± 0.15 |   6300.16 ± 766.25 |  2.0000 ± 0.0000 |
|   150 |   2.79 ± 0.20 |   6516.72 ± 355.71 |  2.0000 ± 0.0000 |
|   180 |   3.29 ± 0.22 |   5776.46 ± 824.25 |  2.0000 ± 0.0000 |
|   210 |   4.20 ± 0.20 |   5534.97 ± 451.44 |  2.0064 ± 0.0055 |
|   240 |   4.37 ± 0.24 |   5168.41 ± 1026.30 |  2.0571 ± 0.0321 |
|   270 |   5.02 ± 0.20 |   3999.37 ± 519.99 |  2.0234 ± 0.0186 |
|   300 |   6.20 ± 0.27 |   1698.16 ± 240.99 |  2.0343 ± 0.0298 |

Mean 3.26 ± 0.13 ms/frame over the five runs — slightly faster than the serial
sweep overall, and ~1.9x at the dense final frame (11.8 → 6.2 ms) where there is
the most independent work to fan out. Early frames carry fixed worker-dispatch
overhead that dominates while there is little to solve, so the speed-up is
concentrated in the dense late frames. Penetration holds near 2 throughout,
matching the serial sweep — the 8 sub-steps keep the slab decomposition as tight
as the serial order.

### Snapshots

The benchmark can render selected frames through a pyglet window with
`--snapshot`, or encode the whole run to an mp4 with `--video` (needs ffmpeg).
Below, three stages of the streamed drop: a few early bodies, several distinct
piles, and the final merged pile.

| Frame 40 (singletons) | Frame 150 (distinct islands) | Frame 300 (settled) |
|:---:|:---:|:---:|
| ![drop box, frame 40](docs/images/drop_box_frame0040.png) | ![drop box, frame 150](docs/images/drop_box_frame0150.png) | ![drop box, frame 300](docs/images/drop_box_frame0300.png) |

## Documentation

bocphysics is built on [`bocpy`](https://pypi.org/project/bocpy/), a library
for **Behavior-Oriented Concurrency**: data lives inside *cowns* and code runs
as *behaviors* that the scheduler dispatches once the cowns they need are
available, eliminating data races and deadlocks by construction.

A full tutorial — rigid-body physics, the engine's position-based (XPBD) solver,
and the step-by-step conversion to a parallel BOC solver — along with a Sphinx
API reference is in development under `docs/`.

## Development

```bash
pip install -e .[test,lint]
pytest                 # run the test suite
flake8 src test bench scripts
```

## License

See [LICENSE](LICENSE).
