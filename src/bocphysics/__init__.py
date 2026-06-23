"""2D Rigid Physics Simulation.

This version of the physics simulation provides an unoptimised, easier
to understand version of the system. Many of the algorithms are written
to accommodate any convex polygon.
"""

from argparse import ArgumentParser

from .config import DetectionKind, Resolution
from .engine import PhysicsMode
from .scene import make_pyramid_scene, make_stack_scene, Scene

# scenes whose body count is set by a --levels row count
PARAMETRIC_SCENES = {"stack": make_stack_scene, "pyramid": make_pyramid_scene}


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
    args = parser.parse_args()

    # imported here, not at module scope, so importing bocphysics in a worker
    # sub-interpreter never pulls pyglet (and its X11 window) into that worker
    from .simulation import Simulation

    simulation = Simulation(resolution=Resolution.from_string(args.size),
                            physics_mode=PhysicsMode[args.mode.upper()],
                            detection_kind=DetectionKind[args.detect.upper()],
                            debug=args.debug,
                            show_contacts=args.show_contacts,
                            snapshot=args.snapshot,
                            scene=load_scene(args.scene, args.levels),
                            parallel=args.parallel,
                            workers=args.workers)
    simulation.run()
