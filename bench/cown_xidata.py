"""Timing probe for a cown carrying a tuple of matrices through XIData.

A value stored in a :class:`~bocpy.Cown` lives in a proto-region as serialized
XIData. :meth:`Cown.acquire` deserializes that payload into the caller's
interpreter; :meth:`Cown.release` serializes the (possibly mutated) contents
back. This probe puts a tuple of ``count`` matrices in a cown and times the
acquire (deserialize) and release (serialize) halves of one round trip,
reporting how the cost scales with the number of matrices. Pass
``--container namedtuple`` to wrap the matrices in a NamedTuple instead of a
plain tuple and compare the behavior.

Run from the repo root with the project venv active::

    python bench/cown_xidata.py --counts 1,2,4,8,16,32 --rows 32 --cols 32
"""

import argparse
import statistics
import time
from typing import NamedTuple, Tuple

from bocpy import Cown, Matrix


class MatrixBundle(NamedTuple):
    """A NamedTuple wrapper carrying a tuple of matrices as a single field."""

    matrices: Tuple[Matrix, ...]


def make_payload(count: int, rows: int, cols: int, container: str):
    """Build ``count`` randomized matrices wrapped in the requested container."""
    matrices = tuple(Matrix.uniform(-1.0, 1.0, size=(rows, cols)) for _ in range(count))
    return MatrixBundle(matrices) if container == "namedtuple" else matrices


def time_round_trip(payload: tuple) -> tuple:
    """Time one acquire (deserialize) and release (serialize) of a fresh cown."""
    cown = Cown(payload)
    start = time.perf_counter()
    cown.acquire()
    mid = time.perf_counter()
    cown.release()
    end = time.perf_counter()
    return (mid - start) * 1e6, (end - mid) * 1e6


def measure(count: int, rows: int, cols: int, trials: int, warmup: int, container: str) -> tuple:
    """Return mean/std acquire and release microseconds over ``trials`` round trips."""
    for _ in range(warmup):
        time_round_trip(make_payload(count, rows, cols, container))

    acquires = []
    releases = []
    for _ in range(trials):
        acquire_us, release_us = time_round_trip(make_payload(count, rows, cols, container))
        acquires.append(acquire_us)
        releases.append(release_us)

    return mean_std(acquires), mean_std(releases)


def mean_std(values: list) -> tuple:
    """Return the mean and sample standard deviation of a sequence of values."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def run(counts: list, rows: int, cols: int, trials: int, warmup: int, seed: int, container: str):
    """Measure the acquire/release cost for each matrix count and print a table."""
    Matrix.seed(seed)
    print(f"container={container} rows={rows} cols={cols} "
          f"trials={trials} warmup={warmup} seed={seed}")
    print(f"\n{'matrices':>8} {'acquire us':>18} {'release us':>18} "
          f"{'acq/mat us':>12} {'rel/mat us':>12}")

    for count in counts:
        (acq_mean, acq_std), (rel_mean, rel_std) = measure(count, rows, cols, trials, warmup, container)
        print(f"{count:>8} {acq_mean:>8.2f} \u00b1 {acq_std:>6.2f} "
              f"{rel_mean:>8.2f} \u00b1 {rel_std:>6.2f} "
              f"{acq_mean / count:>12.3f} {rel_mean / count:>12.3f}")


def parse_counts(text: str) -> list:
    """Parse a comma-separated list of positive matrix counts."""
    counts = [int(part) for part in text.split(",") if part.strip()]
    if not counts or any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("counts must be a comma-separated list of positive ints")
    return counts


def main():
    """Parse arguments and run the cown XIData timing probe."""
    parser = argparse.ArgumentParser(description="Cown XIData acquire/release timing probe")
    parser.add_argument("--counts", type=parse_counts, default=[1, 2, 4, 8, 16, 32, 64],
                        help="Comma-separated matrix counts to sweep")
    parser.add_argument("--rows", type=int, default=32, help="Rows per matrix")
    parser.add_argument("--cols", type=int, default=32, help="Columns per matrix")
    parser.add_argument("--trials", type=int, default=200, help="Timed round trips per count")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed round trips per count")
    parser.add_argument("--seed", type=int, default=0, help="Matrix PRNG seed")
    parser.add_argument("--container", choices=["tuple", "namedtuple"], default="tuple",
                        help="Container wrapping the matrices inside the cown")
    args = parser.parse_args()

    run(args.counts, args.rows, args.cols, args.trials, args.warmup, args.seed, args.container)


if __name__ == "__main__":
    main()
