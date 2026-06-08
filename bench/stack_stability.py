"""Headless stack-stability benchmark for the physics engine.

Builds a deterministic vertical stack of equal boxes on a floor, lets it
settle, and reports the metrics that position correction actually governs:

* sink        -- how far the bottom box sinks below its ideal resting height
* penetration -- total overlap depth across all contacts at rest
* lean        -- largest absolute body angle (a stable stack stays upright)
* drift       -- largest horizontal displacement from the stack centre line
* jitter      -- mean kinetic energy over the final frames (rest should be still)

Unlike the drop-box probe this scene is fully reproducible: there is no
``Matrix.uniform``, so the same parameters give the same numbers every run.

Run from the repo root with the project venv active::

    python bench/stack_stability.py --levels 8 --frames 600
"""

import argparse

from bocpy import Matrix

from bocphysics.bodies import Polygon
from bocphysics.collisions import detect_collision
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine

BOX_SIZE = 2.0
FLOOR_TOP = 9.0


def build_stack(engine, levels: int):
    """Add a floor and a vertical column of boxes resting on it."""
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    # stack upward in the y-down world: the lowest box centre sits one half
    # height above the floor top, each box one full height above the last
    for i in range(levels):
        y = FLOOR_TOP - BOX_SIZE / 2 - i * BOX_SIZE
        box = Polygon.create_rectangle(BOX_SIZE, BOX_SIZE, 2.0,
                                       (50, 120, 200))
        engine.add_body(box.move_to(Matrix.vector([0, y])))


def build_pyramid(engine, levels: int):
    """Add a floor and a brick pyramid where upper boxes span two below.

    Description:
        Each box above the base rests on the seam between two lower boxes, so
        it is supported by two offset contacts. This torque-prone layout is the
        case where full position projection can over-correct and topple a
        stack, unlike a perfectly collinear column.
    """
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    for row in range(levels):
        y = FLOOR_TOP - BOX_SIZE / 2 - row * BOX_SIZE
        count = levels - row
        # centre each row so the box above straddles two boxes below
        offset = (count - 1) * BOX_SIZE / 2
        for col in range(count):
            x = col * BOX_SIZE - offset
            box = Polygon.create_rectangle(BOX_SIZE, BOX_SIZE, 2.0,
                                           (50, 120, 200))
            engine.add_body(box.move_to(Matrix.vector([x, y])))


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


def dynamic_bodies(engine):
    """Return the engine's dynamic bodies in insertion order."""
    return [body for body in engine.bodies if body.physics]


def measure(levels: int, frames: int, dt: float, tail: int, layout: str = "column") -> dict:
    """Settle a stack and return its stability metrics."""
    engine = PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                           DetectionKind.QUADTREE, show_contacts=False)
    if layout == "pyramid":
        build_pyramid(engine, levels)
    else:
        build_stack(engine, levels)

    # record each body's starting position so drift is measured against it
    start_x = {id(body): body.position.x for body in dynamic_bodies(engine)}

    jitter_sum = 0.0
    for frame in range(1, frames + 1):
        engine.step(dt)
        if frame > frames - tail:
            jitter_sum += total_kinetic_energy(engine)

    bodies = dynamic_bodies(engine)
    ideal_bottom = FLOOR_TOP - BOX_SIZE / 2
    bottom = max(bodies, key=lambda b: b.position.y)
    return {
        "sink": bottom.position.y - ideal_bottom,
        "penetration": total_penetration(engine),
        "lean": max(abs(b.angle) for b in bodies),
        "drift": max(abs(b.position.x - start_x[id(b)]) for b in bodies),
        "jitter": jitter_sum / tail,
    }


def main():
    """Parse arguments, settle one stack, and print its stability metrics."""
    parser = argparse.ArgumentParser(description="Stack-stability benchmark.")
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--dt", type=float, default=1 / 60)
    parser.add_argument("--tail", type=int, default=60)
    parser.add_argument("--layout", choices=("column", "pyramid"), default="column")
    args = parser.parse_args()

    metrics = measure(args.levels, args.frames, args.dt, args.tail, args.layout)
    print(f"layout={args.layout} levels={args.levels} frames={args.frames} "
          f"dt={args.dt} tail={args.tail}")
    print(f"  sink        {metrics['sink']:+.4f}  (below ideal resting height)")
    print(f"  penetration {metrics['penetration']:.4f}")
    print(f"  lean        {metrics['lean']:.4f}  rad")
    print(f"  drift       {metrics['drift']:.4f}  (from start position)")
    print(f"  jitter      {metrics['jitter']:.4f}  mean KE over last {args.tail} frames")


if __name__ == "__main__":
    main()
