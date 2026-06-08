"""Headless drop-box benchmark for the physics engine.

Drops a variety of shapes into an open box and runs the engine without a
window, reporting wall-clock cost per frame and two convergence proxies
(total kinetic energy and total penetration depth). This is a perf /
convergence probe, NOT a reproducible correctness test: spawn placement draws
from an unseeded ``random``, so numbers vary run to run.

Run from the repo root with the project venv active::

    python bench/drop_box.py --shapes 80 --frames 300
"""

import argparse
import math
import os
import random
import statistics
import time

from bocpy import Matrix

from bocphysics.bodies import Circle, Polygon
from bocphysics.collisions import detect_collision
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.scene import OPEN_BOX


def spawn_one(engine):
    """Drop a single randomly-shaped, randomly-rotated body high above the floor."""
    x = random.uniform(-11, 11)
    # the world is y-down with the floor near y=10, so small y is high up
    y = random.uniform(-13, -7)
    angle = random.uniform(0, 2 * math.pi)
    color = (random.randint(40, 255), random.randint(40, 255), random.randint(40, 255))
    kind = random.random()
    if kind < 0.4:
        body = Circle.create(random.uniform(0.6, 1.2), 2.0, color)
    elif kind < 0.7:
        body = Polygon.create_rectangle(random.uniform(1.2, 2.4),
                                        random.uniform(1.2, 2.4), 2.0, color)
    else:
        body = Polygon.create_regular_polygon(random.randint(3, 8),
                                              random.uniform(0.8, 1.4), 2.0, color)

    engine.add_body(body.move_to(Matrix.vector([x, y])).rotate_to(angle))


def make_spawn_schedule(shapes: int, frames: int, spawn_frames: int):
    """Spread the shape drops over a window so islands grow from singletons."""
    schedule = [0] * (frames + 1)
    if shapes <= 0:
        return schedule

    # a window of 0 means drop everything at once on the first frame
    window = min(max(spawn_frames, 1), frames)
    for i in range(shapes):
        schedule[1 + i * window // shapes] += 1

    return schedule


def total_kinetic_energy(engine) -> float:
    """Sum translational and rotational kinetic energy over dynamic bodies."""
    energy = 0.0
    for body in engine.bodies:
        if body.physics:
            energy += 0.5 * body.mass * body.linear_velocity.magnitude_squared()
            energy += 0.5 * body.inertia * body.angular_velocity**2

    return energy


def total_penetration(engine) -> float:
    """Measure total penetration depth without resolving any collisions."""
    pairs = []
    engine.detection.find_all_intersections(engine.bodies, pairs)
    depth = 0.0
    for a, b in pairs:
        collision = detect_collision(a, b)
        if collision is not None:
            depth += collision.depth

    return depth


def make_camera(engine):
    """Build a camera matching the engine's scale and centre for rendering."""
    from bocphysics.render import Camera

    return Camera(engine.center, engine.scale, engine.height * engine.scale)


def save_snapshot(window, engine, camera, path: str):
    """Render the current engine state into the window and save it as a PNG."""
    import pyglet

    window.switch_to()
    window.clear()
    batch = pyglet.graphics.Batch()
    kept = engine.draw(batch, camera)
    batch.draw()
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    buffer.save(path)
    return kept


def open_encoder(path: str, width: int, height: int, fps: int):
    """Start an ffmpeg subprocess that consumes raw RGBA frames over stdin."""
    import subprocess

    return subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "warning",
         "-f", "rawvideo", "-pix_fmt", "rgba",
         "-s", f"{width}x{height}", "-r", str(fps),
         "-i", "-", "-vf", "vflip",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE)


def record_video(shapes: int, frames: int, dt: float, mode: str, detect: str,
                 spawn_frames: int, path: str, fps: int):
    """Run one simulation, rendering every frame and encoding it to a video file."""
    import pyglet

    from bocphysics.engine import PhysicsEngine

    engine = PhysicsEngine(1200, 900, PhysicsMode[mode.upper()],
                           DetectionKind[detect.upper()], show_contacts=False)
    for body in OPEN_BOX.build():
        engine.add_body(body)

    schedule = make_spawn_schedule(shapes, frames, spawn_frames)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    window = pyglet.window.Window(1200, 900, "bocphysics benchmark", visible=False)
    pyglet.gl.glClearColor(1, 1, 1, 1)
    camera = make_camera(engine)

    # query the real framebuffer size, which can differ from the window on HiDPI
    window.switch_to()
    window.clear()
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    encoder = open_encoder(path, buffer.width, buffer.height, fps)
    try:
        for frame in range(1, frames + 1):
            for _ in range(schedule[frame]):
                spawn_one(engine)

            engine.step(dt)
            window.switch_to()
            window.clear()
            batch = pyglet.graphics.Batch()
            kept = engine.draw(batch, camera)
            batch.draw()
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            data = buffer.get_image_data().get_data("RGBA", buffer.width * 4)
            encoder.stdin.write(data)
            del kept
    finally:
        encoder.stdin.close()
        encoder.wait()
        window.close()

    print(f"wrote {path} ({frames} frames at {fps} fps, {len(engine.bodies)} bodies)")


def simulate(shapes: int, frames: int, dt: float, mode: str, detect: str, report: int,
             spawn_frames: int, snapshot_frames=(), snapshot_dir="docs/images"):
    """Run one simulation, returning report rows, mean ms/frame, and body count."""
    from bocphysics.engine import PhysicsEngine

    engine = PhysicsEngine(1200, 900, PhysicsMode[mode.upper()],
                           DetectionKind[detect.upper()], show_contacts=False)
    for body in OPEN_BOX.build():
        engine.add_body(body)

    schedule = make_spawn_schedule(shapes, frames, spawn_frames)

    window = None
    camera = None
    if snapshot_frames:
        import pyglet

        os.makedirs(snapshot_dir, exist_ok=True)
        window = pyglet.window.Window(1200, 900, "bocphysics benchmark", visible=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        camera = make_camera(engine)

    rows = []
    total_elapsed = 0.0
    interval_elapsed = 0.0
    for frame in range(1, frames + 1):
        for _ in range(schedule[frame]):
            spawn_one(engine)

        start = time.perf_counter()
        engine.step(dt)
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed
        interval_elapsed += elapsed

        if window is not None and frame in snapshot_frames:
            path = os.path.join(snapshot_dir, f"drop_box_frame{frame:04d}.png")
            save_snapshot(window, engine, camera, path)
            print(f"  saved snapshot {path}")

        if frame % report == 0:
            rows.append((frame, interval_elapsed / report * 1000,
                         total_kinetic_energy(engine), total_penetration(engine)))
            interval_elapsed = 0.0

    if window is not None:
        window.close()

    return rows, total_elapsed / frames * 1000, len(engine.bodies)


def mean_std(values):
    """Return the mean and sample standard deviation of a sequence of values."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def run(shapes: int, frames: int, dt: float, mode: str, detect: str, report: int,
        runs: int, spawn_frames: int, snapshot_frames=(), snapshot_dir="docs/images"):
    """Run the benchmark over several runs and print mean +/- std statistics."""
    print(f"shapes={shapes} frames={frames} dt={dt} mode={mode} detect={detect} "
          f"runs={runs} spawn_frames={spawn_frames}")

    all_rows = []
    mean_ms_values = []
    body_count = 0
    for run_index in range(runs):
        # only the first run captures snapshots so we do not overwrite them
        frames_to_snap = snapshot_frames if run_index == 0 else ()
        rows, mean_ms, body_count = simulate(shapes, frames, dt, mode, detect, report,
                                             spawn_frames, frames_to_snap, snapshot_dir)
        mean_ms_values.append(mean_ms)
        all_rows.append(rows)

    print(f"\n{'frame':>6} {'ms/frame':>16} {'kinetic':>22} {'penetration':>20}")
    for i in range(len(all_rows[0])):
        frame = all_rows[0][i][0]
        ms_mean, ms_std = mean_std([all_rows[r][i][1] for r in range(runs)])
        kin_mean, kin_std = mean_std([all_rows[r][i][2] for r in range(runs)])
        pen_mean, pen_std = mean_std([all_rows[r][i][3] for r in range(runs)])
        print(f"{frame:>6} {ms_mean:>8.2f} \u00b1 {ms_std:>5.2f} "
              f"{kin_mean:>11.2f} \u00b1 {kin_std:>8.2f} "
              f"{pen_mean:>8.4f} \u00b1 {pen_std:>6.4f}")

    ms_mean, ms_std = mean_std(mean_ms_values)
    print(f"\nmean {ms_mean:.3f} \u00b1 {ms_std:.3f} ms/frame over {runs} runs  "
          f"bodies={body_count}")


def main():
    """Parse arguments and run the drop-box benchmark."""
    parser = argparse.ArgumentParser(description="Headless drop-box physics benchmark")
    parser.add_argument("--shapes", type=int, default=80, help="Number of dynamic shapes to drop")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames to simulate")
    parser.add_argument("--dt", type=float, default=1 / 60, help="Time step per frame in seconds")
    parser.add_argument("--mode", default="friction",
                        choices=["none", "basic", "rotation", "friction"])
    parser.add_argument("--detect", default="quadtree", choices=["quadtree", "basic"])
    parser.add_argument("--report", type=int, default=30, help="Frames between report lines")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average over")
    parser.add_argument("--spawn-frames", type=int, default=-1,
                        help="Frames over which to stream the drops (default ~70%% of --frames)")
    parser.add_argument("--snapshot", default="",
                        help="Comma-separated frame numbers to capture as PNG snapshots")
    parser.add_argument("--snapshot-dir", default="docs/images",
                        help="Directory to write snapshot PNGs into")
    parser.add_argument("--video", default="",
                        help="Render every frame and encode an mp4 at this path (needs ffmpeg)")
    parser.add_argument("--fps", type=int, default=60, help="Frame rate for --video output")
    args = parser.parse_args()

    spawn_frames = args.spawn_frames if args.spawn_frames >= 0 else int(args.frames * 0.7)
    if args.video:
        record_video(args.shapes, args.frames, args.dt, args.mode, args.detect,
                     spawn_frames, args.video, args.fps)
        return

    snapshot_frames = frozenset(int(f) for f in args.snapshot.split(",") if f.strip())
    run(args.shapes, args.frames, args.dt, args.mode, args.detect, args.report,
        args.runs, spawn_frames, snapshot_frames, args.snapshot_dir)


if __name__ == "__main__":
    main()
