"""Measure the BOC behavior-graph width and depth of a parallel frame.

The parallel stepper emits, per frame, ``num_substeps`` layers of one intra
behavior per patch plus the colored seam behaviors, then a single writeback.
Under per-cown FIFO with unbounded workers the intra layer is one critical-path
level (all patches sit on disjoint cowns) and each seam colour is one more level
(a colour is an independent set of patch pairs), so the span is::

    span = num_substeps * (1 + seam_colors) + 1

This tool advances a drop-box shower with the serial engine to build a realistic
pile, then at report frames reconstructs that exact schedule and computes the
ASAP level of every behavior. Depth is the critical path; max and average width
are the available concurrency. No workers run -- it is pure structural analysis
of the cown graph, so it needs no display and is deterministic under --seed.

Run from the repo root with the project venv active::

    python bench/boc_width.py --shapes 160 --frames 300 --seed 7 --sweep
"""

import argparse
from collections import Counter, defaultdict

from bocpy import Matrix
from drop_box import make_spawn_schedule, spawn_one

from bocphysics import parallel
from bocphysics.config import DetectionKind
from bocphysics.engine import PhysicsEngine
from bocphysics.parallel import (colored_seam_order, ParallelStepper,
                                 seam_groups)
from bocphysics.scene import OPEN_BOX


def seam_colors(keys, num_patches):
    """Return {key: colour} using the same greedy edge-colouring as the scheduler."""
    used = [set() for _ in range(num_patches)]
    colors = {}
    for (i, j) in sorted(keys):
        c = 0
        while c in used[i] or c in used[j]:
            c += 1
        colors[(i, j)] = c
        used[i].add(c)
        used[j].add(c)
    return colors


def schedule_levels(num_patches, seam_order, num_substeps):
    """ASAP level of every behavior under per-cown FIFO with unbounded workers.

    Returns (depth, level_counts) where level_counts[L] is how many behaviors can
    run concurrently at critical-path level L. Intra behaviors lock one patch
    cown; seam behaviors lock two; the final writeback locks every patch.
    """
    last = defaultdict(int)
    levels = []
    for _ in range(num_substeps):
        for p in range(num_patches):
            last[p] += 1
            levels.append(last[p])
        for (i, j) in seam_order:
            lvl = max(last[i], last[j]) + 1
            last[i] = last[j] = lvl
            levels.append(lvl)
    levels.append(max(last[p] for p in range(num_patches)) + 1)
    return max(levels), Counter(levels)


def analyze(stepper, frame):
    """Recompute the frame's partition and print its behavior-graph width/depth."""
    engine = stepper.engine
    engine.contacts.clear()
    engine.update_swept_aabbs(stepper.dt)
    engine.collisions.clear()
    engine.broad_phase()
    partition = stepper.cut_partition()
    if not partition.patches:
        print(f"frame {frame:>4}: empty partition")
        return

    num_patches = len(partition.patches)
    sizes = sorted((len(p.bodies) for p in partition.patches), reverse=True)
    keys = list(seam_groups(partition))
    order = colored_seam_order(keys, num_patches)
    colors = seam_colors(keys, num_patches)
    num_colors = (max(colors.values()) + 1) if colors else 0

    depth, counts = schedule_levels(num_patches, order, num_substeps=engine.num_substeps)
    total = sum(counts.values())
    max_width = max(counts.values())
    imbalance = sizes[0] / (sum(sizes) / num_patches)

    print(f"frame {frame:>4}: bodies={sum(sizes):>3}  patches={num_patches:>2} "
          f"(maxsize={sizes[0]:>2} imbalance={imbalance:.2f}x)  seams={len(keys):>2} "
          f"colors={num_colors}  span={depth:>2} max_width={max_width:>2} "
          f"work/span={total / depth:.2f}")


def report_frames(report, frames):
    """Frame numbers to analyze: every ``report`` frames plus the final frame."""
    marks = set(range(report, frames + 1, report))
    marks.add(frames)
    return marks


def main():
    """Parse arguments, advance the shower, and report the graph metrics."""
    parser = argparse.ArgumentParser(description="BOC behavior-graph width/depth probe")
    parser.add_argument("--shapes", type=int, default=160, help="Dynamic shapes to stream in")
    parser.add_argument("--frames", type=int, default=300, help="Frames to simulate")
    parser.add_argument("--seed", type=int, default=7, help="Seed for the Matrix PRNG")
    parser.add_argument("--slabs", type=int, default=parallel.DEFAULT_SLABS,
                        help="Equal-population vertical slabs to cut")
    parser.add_argument("--report", type=int, default=60, help="Frames between report lines")
    parser.add_argument("--sweep", action="store_true",
                        help="After the run, sweep slab counts on the frozen final frame")
    args = parser.parse_args()

    Matrix.seed(args.seed)
    engine = PhysicsEngine(1200, 900, DetectionKind.QUADTREE, show_contacts=False)
    for body in OPEN_BOX.build():
        engine.add_body(body)
    schedule = make_spawn_schedule(args.shapes, args.frames, int(args.frames * 0.7))

    stepper = ParallelStepper(engine, num_slabs=args.slabs)
    stepper.dt = 1 / 60
    marks = report_frames(args.report, args.frames)

    for frame in range(1, args.frames + 1):
        for _ in range(schedule[frame]):
            spawn_one(engine)
        if frame in marks:
            analyze(stepper, frame)
        engine.step(stepper.dt)

    if args.sweep:
        print(f"\nSlab sweep on the frozen final frame ({args.frames}):")
        for slabs in (4, 6, 8, 10, 12, 16, 20, 24):
            stepper.num_slabs = slabs
            analyze(stepper, args.frames)


if __name__ == "__main__":
    main()
