"""2D Rigid Physics Simulation.

This version of the physics simulation provides an unoptimised, easier
to understand version of the system. Many of the algorithms are written
to accommodate any convex polygon.
"""

from argparse import ArgumentParser

from .config import DetectionKind, Resolution
from .engine import PhysicsMode
from .scene import Scene
from .simulation import Simulation


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
    args = parser.parse_args()

    simulation = Simulation(resolution=Resolution.from_string(args.size),
                            physics_mode=PhysicsMode[args.mode.upper()],
                            detection_kind=DetectionKind[args.detect.upper()],
                            debug=args.debug,
                            show_contacts=args.show_contacts,
                            snapshot=args.snapshot,
                            scene=Scene.load(args.scene))
    simulation.run()
