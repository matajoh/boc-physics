"""2D rigid-body physics engine built on Behavior-Oriented Concurrency.

bocphysics simulates convex polygons and circles under gravity, collision,
and friction, running the contact solver in parallel on bocpy cowns and
behaviors. It doubles as a teaching aid for programming with BOC.
"""

from argparse import ArgumentParser

from .config import DetectionKind, Resolution
from .engine import PhysicsMode
from .scene import make_pachinko_scene, make_pyramid_scene, make_stack_scene, Scene

PARAMETRIC_SCENES = {"stack": make_stack_scene, "pyramid": make_pyramid_scene,
                     "pachinko": make_pachinko_scene}


def load_scene(name: str, levels):
    """Load a scene by name, re-parametrising stack/pyramid when --levels is given."""
    if levels is not None and name in PARAMETRIC_SCENES:
        return PARAMETRIC_SCENES[name](levels)

    return Scene.load(name)


def main():
    """Parse command-line arguments and run the simulation."""
    parser = ArgumentParser(description="2D Rigid Physics Simulation")
    parser.add_argument("--mode", "-m", help="Physics mode to use",
                        choices=["basic", "friction", "none", "rotation"],
                        default="friction")
    parser.add_argument("--show-contacts", action="store_true", help="Show contact points")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--snapshot", "-ss", action="store_true", help="Save a snapshot of the simulation")
    parser.add_argument("--size", "-s", help="Size of the window", default="1200x900")
    parser.add_argument("--width", type=int, default=None,
                        help="Window width in pixels (overrides the width in --size)")
    parser.add_argument("--height", type=int, default=None,
                        help="Window height in pixels (overrides the height in --size)")
    parser.add_argument("--detect", type=str, help="Detection algorithm to use",
                        choices=["quadtree", "basic"], default="quadtree")
    parser.add_argument("--scene", type=str, help="Built-in scene name or path to a scene JSON file",
                        default="default")
    parser.add_argument("--levels", type=int, default=None,
                        help="Row count for the parametric 'stack' and 'pyramid' scenes")
    parser.add_argument("--parallel", action="store_true",
                        help="Run the physics step across BOC worker sub-interpreters")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker count for --parallel (default: auto)")
    parser.add_argument("--batched", action="store_true",
                        help="Use the colour-batched velocity kernel instead of the scalar solver")
    parser.add_argument("--overlay", choices=["none", "slabs", "quadtree"], default="none",
                        help="Draw a partition overlay: equal-population slabs or quadtree cells")
    parser.add_argument("--video", default="",
                        help="Record frames to an mp4 at this path instead of running live (needs ffmpeg)")
    parser.add_argument("--fps", type=int, default=60, help="Frame rate for --video output")
    parser.add_argument("--frames", type=int, default=600,
                        help="Number of frames to record when --video is given")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the Matrix PRNG for a reproducible run")
    args = parser.parse_args()

    from .simulation import Simulation

    if args.seed is not None:
        from bocpy import Matrix
        Matrix.seed(args.seed)

    from . import solver
    # Snapshotted by ParallelStepper.begin(); must be set before the engine starts stepping.
    solver.use_batched_solver = args.batched

    resolution = Resolution.from_string(args.size)
    if args.width is not None:
        resolution = resolution._replace(width=args.width)
    if args.height is not None:
        resolution = resolution._replace(height=args.height)

    simulation = Simulation(resolution=resolution,
                            physics_mode=PhysicsMode[args.mode.upper()],
                            detection_kind=DetectionKind[args.detect.upper()],
                            debug=args.debug,
                            show_contacts=args.show_contacts,
                            snapshot=args.snapshot,
                            scene=load_scene(args.scene, args.levels),
                            parallel=args.parallel,
                            workers=args.workers,
                            overlay=args.overlay,
                            visible=not args.video)
    if args.video:
        simulation.record(args.video, args.frames, args.fps)
    else:
        simulation.run()
