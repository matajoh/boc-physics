"""Track drop_box ms/frame across commits into REFACTOR.md.

Runs one headless drop_box round and samples ms/frame at four frame
checkpoints (25/50/75/100% of the run). Because bodies accumulate over frames
via the spawn schedule, frame count is a proxy for body count, so each
checkpoint reflects a heavier scene. Every invocation contributes one point per
checkpoint for the current commit, letting you watch the four curves move as a
refactor progresses.

The Measurements table in REFACTOR.md is the source of truth; the mermaid chart
is regenerated from it on every run. Run from the repo root with the project
venv active::

    python scripts/track_refactor.py --shapes 80 --frames 300 --seed 7
"""

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO_ROOT, "bench")
REFACTOR_PATH = os.path.join(REPO_ROOT, "REFACTOR.md")

CHECKPOINTS = (25, 50, 75, 100)


def load_simulate():
    """Import the drop_box benchmark's simulate() without packaging bench/."""
    if BENCH_DIR not in sys.path:
        sys.path.insert(0, BENCH_DIR)

    from drop_box import simulate

    return simulate


def git_output(*args) -> str:
    """Return trimmed stdout from a git command run in the repo root."""
    result = subprocess.run(["git", *args], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True)
    return result.stdout.strip()


def current_commit():
    """Return (label, subject, date) for HEAD, marking a dirty tree with -dirty."""
    short = git_output("rev-parse", "--short", "HEAD")
    subject = git_output("log", "-1", "--format=%s")
    date = git_output("log", "-1", "--format=%cs")
    dirty = bool(git_output("status", "--porcelain"))
    label = f"{short}-dirty" if dirty else short
    return label, subject, date


def measure(simulate, shapes, frames, dt, detect, spawn_frames, seed, runs):
    """Return mean ms/frame at each checkpoint, averaged over the given runs."""
    report = max(1, frames // len(CHECKPOINTS))
    totals = [0.0] * len(CHECKPOINTS)
    body_count = 0
    for run_index in range(runs):
        rows, _mean_ms, body_count = simulate(shapes, frames, dt, detect, report,
                                              spawn_frames, seed + run_index)
        if len(rows) < len(CHECKPOINTS):
            raise SystemExit(f"expected {len(CHECKPOINTS)} report rows, got {len(rows)}; "
                             "pick --frames divisible by 4")
        for i in range(len(CHECKPOINTS)):
            totals[i] += rows[i][1]

    return [total / runs for total in totals], body_count


def parse_table(text):
    """Return ordered (commit, subject, date, [ms...]) rows from REFACTOR.md."""
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 3 + len(CHECKPOINTS):
            continue

        if cells[0] in ("Commit", "---") or set(cells[0]) == {"-"}:
            continue

        try:
            values = [float(cell) for cell in cells[3:]]
        except ValueError:
            continue

        rows.append((cells[0], cells[1], cells[2], values))

    return rows


def upsert_row(rows, commit, subject, date, values):
    """Replace the row for this commit if present, else append it."""
    for index, existing in enumerate(rows):
        if existing[0] == commit:
            rows[index] = (commit, subject, date, values)
            return rows

    rows.append((commit, subject, date, values))
    return rows


def render_chart(rows):
    """Render a mermaid xychart with one line per commit across the checkpoints."""
    labels = ", ".join(f'"{pct}%"' for pct in CHECKPOINTS)
    lines = []
    for row in rows:
        series = ", ".join(f"{value:.3f}" for value in row[3])
        lines.append(f"    line [{series}]")

    ymax = max((max(row[3]) for row in rows), default=1.0)
    y_top = max(1.0, ymax * 1.15)
    body = "\n".join(lines)
    return ("```mermaid\n"
            "xychart-beta\n"
            '    title "drop_box ms/frame per commit"\n'
            f'    x-axis "body-count checkpoint" [{labels}]\n'
            f'    y-axis "ms/frame" 0 --> {y_top:.3f}\n'
            f"{body}\n"
            "```")


def render_table(rows):
    """Render the Measurements table with one row per tracked commit."""
    header = "| Commit | Subject | Date | " + " | ".join(f"{pct}%" for pct in CHECKPOINTS) + " |"
    divider = "| --- | --- | --- | " + " | ".join("---" for _ in CHECKPOINTS) + " |"
    body = []
    for commit, subject, date, values in rows:
        cells = " | ".join(f"{value:.3f}" for value in values)
        body.append(f"| {commit} | {subject} | {date} | {cells} |")

    return "\n".join([header, divider, *body])


def render_document(rows, config):
    """Assemble the full REFACTOR.md contents from the tracked rows."""
    return "\n".join([
        "# Refactor Performance Tracker",
        "",
        "Each commit is one curve: ms/frame sampled at 25/50/75/100% of the drop_box",
        "run. Because bodies accumulate over frames, later checkpoints carry heavier",
        "scenes, so a curve reads left-to-right as cost against a rising body count.",
        "Curves follow the Measurements table order below. Regenerate with:",
        "",
        "```bash",
        f"python scripts/track_refactor.py --shapes {config['shapes']} "
        f"--frames {config['frames']} --seed {config['seed']} --detect {config['detect']}",
        "```",
        "",
        render_chart(rows),
        "",
        "## Measurements",
        "",
        render_table(rows),
        "",
    ])


def main():
    """Parse arguments, measure the current commit, and update REFACTOR.md."""
    parser = argparse.ArgumentParser(description="Track drop_box ms/frame across commits")
    parser.add_argument("--shapes", type=int, default=80, help="Number of dynamic shapes to drop")
    parser.add_argument("--frames", type=int, default=300,
                        help="Frames to simulate (should be divisible by 4)")
    parser.add_argument("--dt", type=float, default=1 / 60, help="Time step per frame in seconds")
    parser.add_argument("--detect", default="quadtree", choices=["quadtree", "basic"])
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the Matrix PRNG; reproduces the same spawns")
    parser.add_argument("--spawn-frames", type=int, default=-1,
                        help="Frames over which to stream the drops (default ~70%% of --frames)")
    parser.add_argument("--runs", type=int, default=1,
                        help="Rounds to average per checkpoint (default 1)")
    args = parser.parse_args()

    spawn_frames = args.spawn_frames if args.spawn_frames >= 0 else int(args.frames * 0.7)
    simulate = load_simulate()
    values, body_count = measure(simulate, args.shapes, args.frames, args.dt, args.detect,
                                 spawn_frames, args.seed, args.runs)
    commit, subject, date = current_commit()

    existing = ""
    if os.path.exists(REFACTOR_PATH):
        with open(REFACTOR_PATH, encoding="utf-8") as handle:
            existing = handle.read()

    rows = parse_table(existing)
    rows = upsert_row(rows, commit, subject, date, values)
    config = {"shapes": args.shapes, "frames": args.frames,
              "seed": args.seed, "detect": args.detect}
    with open(REFACTOR_PATH, "w", encoding="utf-8") as handle:
        handle.write(render_document(rows, config))

    summary = ", ".join(f"{pct}%={value:.3f}" for pct, value in zip(CHECKPOINTS, values))
    print(f"{commit}: {summary} ms/frame (final bodies={body_count})")
    print(f"wrote {REFACTOR_PATH}")


if __name__ == "__main__":
    main()
