"""Narrow-phase profiling harness for the PLAN-narrow optimisation work.

Builds a dense settled pile by streaming shapes in waves (so spawn-overlap
retries do not silently drop bodies), then cProfiles a fixed measured window and
reports per-frame wall time plus build_contacts' share of the profiled total.
Use it to capture the Step 0 baseline and to re-measure after Tasks A / B / C.

Run from the repo root with the project venv active::

    python bench/profile_narrow.py --target 200 --measure 150 --seed 7
"""

import argparse
import cProfile
import io
import os
import pstats
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench.drop_box as db  # noqa: E402
from bocpy import Matrix  # noqa: E402

from bocphysics.config import DetectionKind  # noqa: E402
from bocphysics.engine import PhysicsEngine  # noqa: E402
from bocphysics.scene import OPEN_BOX  # noqa: E402


def dynamic_count(engine) -> int:
    """Count the dynamic bodies currently in the engine."""
    return sum(1 for body in engine.bodies if body.physics)


def build_pile(target: int, seed: int, wave: int, wave_frames: int,
               settle: int) -> PhysicsEngine:
    """Stream shapes in waves until the pile reaches target bodies, then settle."""
    Matrix.seed(seed)
    engine = PhysicsEngine(1200, 900, DetectionKind.QUADTREE, show_contacts=False)
    for body in OPEN_BOX.build():
        engine.add_body(body)

    while dynamic_count(engine) < target:
        for _ in range(wave):
            db.spawn_one(engine)
        for _ in range(wave_frames):
            engine.step(1 / 60)

    for _ in range(settle):
        engine.step(1 / 60)

    return engine


def profile_steps(engine, measure: int):
    """Profile a measured window of steps; return (ms_per_frame, stats_text)."""
    profiler = cProfile.Profile()
    profiler.enable()
    start = time.perf_counter()
    for _ in range(measure):
        engine.step(1 / 60)

    wall = time.perf_counter() - start
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
    stats.print_stats(16)
    return wall / measure * 1000, stream.getvalue()


def build_contacts_share(stats_text: str) -> float:
    """Return build_contacts' cumtime as a percentage of the profiled total."""
    total = float(re.search(r"in ([0-9.]+) seconds", stats_text).group(1))
    rows = [line for line in stats_text.splitlines()
            if "build_contacts" in line and "xpbd" in line]
    cumtime = float(rows[0].split()[3]) if rows else 0.0
    return cumtime / total * 100 if total else 0.0


def main():
    """Parse arguments and print the narrow-phase profile for one pile size."""
    parser = argparse.ArgumentParser(description="Narrow-phase profiling harness")
    parser.add_argument("--target", type=int, default=200,
                        help="Dynamic-body count to grow the pile to")
    parser.add_argument("--measure", type=int, default=150,
                        help="Frames in the profiled measurement window")
    parser.add_argument("--seed", type=int, default=7, help="Matrix PRNG seed")
    parser.add_argument("--wave", type=int, default=20,
                        help="Shapes dropped per spawn wave")
    parser.add_argument("--wave-frames", type=int, default=25,
                        help="Settle frames between spawn waves")
    parser.add_argument("--settle", type=int, default=120,
                        help="Settle frames after the pile is full")
    parser.add_argument("--full", action="store_true",
                        help="Print the full top-16 tottime table, not just the summary")
    args = parser.parse_args()

    engine = build_pile(args.target, args.seed, args.wave, args.wave_frames,
                        args.settle)
    bodies = dynamic_count(engine)
    ms_per_frame, stats_text = profile_steps(engine, args.measure)
    share = build_contacts_share(stats_text)

    if args.full:
        print(stats_text)

    print(f"seed={args.seed} target={args.target} bodies={bodies} "
          f"measure={args.measure} ms/frame={ms_per_frame:.2f} "
          f"build_contacts={share:.1f}% of profiled total")


if __name__ == "__main__":
    main()
