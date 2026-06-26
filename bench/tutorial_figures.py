"""Render the tutorial's bocphysics-native figures straight from the simulator.

Each figure is produced by the same offscreen render path the live window uses,
so the static-first z-ordering and the partition overlays match exactly what a
reader sees on screen. The hero shot is the generator-fed pachinko scene; the
partition overlays stream a shower of bodies into the default two-ledge scene,
whose asymmetric piles give the slabs and quadtree something interesting to
carve up. Bodies are coloured with the same vivid HLS hue the live window and
the pachinko generator use, so every figure shares one palette. The shower draws
from the seeded Matrix PRNG, so each figure reproduces the same frame every run.

Run from the repo root with the project venv active (an X display is needed even
though the window is invisible)::

    DISPLAY=:0 python bench/tutorial_figures.py
"""

import argparse
from colorsys import hls_to_rgb
import math
import os
from typing import NamedTuple

from bocpy import Matrix

from bocphysics import load_scene
from bocphysics.bodies import Circle, Polygon
from bocphysics.config import Resolution
from bocphysics.engine import PhysicsMode
from bocphysics.simulation import Simulation


class FigureSpec(NamedTuple):
    """One tutorial figure: which scene to settle and how to frame it."""

    name: str
    scene: str
    levels: object
    overlay: str
    frames: int
    resolution: Resolution
    shapes: int
    seed: int


FIGURES = (
    FigureSpec("pachinko_hero", "pachinko", 8, "none", 320, Resolution(900, 900), 0, 0),
    FigureSpec("default_quadtree", "default", None, "quadtree", 280, Resolution(1200, 900), 90, 7),
    FigureSpec("default_slabs", "default", None, "slabs", 280, Resolution(1200, 900), 90, 7),
)


# These spawn helpers mirror drop_box.py on purpose, keeping each bench a standalone teaching script.
def rand_int(low: int, high: int) -> int:
    """Return an integer in [low, high] inclusive from the seeded Matrix PRNG."""
    return min(high, int(Matrix.uniform(low, high + 1)))


def vivid_color():
    """Pick a saturated RGB colour from a random hue, matching the live window."""
    r, g, b = hls_to_rgb(Matrix.uniform(0, 1), 0.5, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def spawn_one(engine):
    """Drop one randomly-shaped, randomly-rotated body high above the floor."""
    x = Matrix.uniform(-11, 11)
    y = Matrix.uniform(-13, -7)
    angle = Matrix.uniform(0, 2 * math.pi)
    color = vivid_color()
    kind = Matrix.uniform(0, 1)
    if kind < 0.4:
        body = Circle.create(Matrix.uniform(0.4, 0.75), 2.0, color)
    elif kind < 0.7:
        body = Polygon.create_rectangle(Matrix.uniform(0.8, 1.5),
                                        Matrix.uniform(0.8, 1.5), 2.0, color)
    else:
        body = Polygon.create_regular_polygon(rand_int(3, 8),
                                              Matrix.uniform(0.55, 0.9), 2.0, color)
    engine.add_body(body.move_to(Matrix.vector([x, y])).rotate_to(angle))


def make_spawn_schedule(shapes: int, frames: int, spawn_frames: int):
    """Spread the shape drops over a window so the piles grow gradually."""
    schedule = [0] * (frames + 1)
    if shapes <= 0:
        return schedule

    window = min(max(spawn_frames, 1), frames)
    for i in range(shapes):
        schedule[1 + i * window // shapes] += 1

    return schedule


def render_figure(figure: FigureSpec, out_dir):
    """Step one scene to a settled frame and save a single PNG of it."""
    Matrix.seed(figure.seed)
    sim = Simulation(resolution=figure.resolution,
                     physics_mode=PhysicsMode.FRICTION,
                     scene=load_scene(figure.scene, figure.levels),
                     overlay=figure.overlay, visible=False)
    schedule = (make_spawn_schedule(figure.shapes, figure.frames, figure.frames // 2)
                if figure.shapes else None)
    for frame in range(1, figure.frames + 1):
        if schedule:
            for _ in range(schedule[frame]):
                spawn_one(sim.engine)
        sim.step_once(1 / 60)

    kept, buffer = sim.render_to_buffer()
    path = os.path.join(out_dir, f"{figure.name}.png")
    buffer.save(path)
    del kept
    sim.close()
    print(f"  saved {path}")


def main():
    """Parse arguments and render every tutorial figure into the output directory."""
    parser = argparse.ArgumentParser(description="Render bocphysics tutorial figures")
    parser.add_argument("--out-dir", default="docs/images",
                        help="Directory to write the figure PNGs into")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for figure in FIGURES:
        render_figure(figure, args.out_dir)


if __name__ == "__main__":
    main()
