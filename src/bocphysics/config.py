"""Module providing configuration data structures for the physics engine.

There is a great advantage to driving simulation software via
configuration files. The first is reproducibility: you can
reproduce a simulation by running the same configuration file. It also
makes it easier to share and collaborate on simulations. Finally, it
makes it easier to experiment with different parameters. While it is
possible to have every possible parameter as a command line argument,
this can get unwieldy very quickly and it is very easy to make mistakes.

The pattern here, of having a structured class (like a NamedTuple) which
parses the JSON and provides default values, is a solid approach to
emulate.
"""

from enum import auto, Enum
from typing import NamedTuple


class Resolution(NamedTuple("Resolution", [("width", int), ("height", int)])):
    """Resolution of the simulation window."""

    @staticmethod
    def from_string(name: str) -> "Resolution":
        """Parse a resolution from a "WIDTHxHEIGHT" string."""
        parts = name.split("x")
        return Resolution(int(parts[0]), int(parts[1]))


class PhysicsMode(Enum):
    """The collision-response model the physics system applies."""

    NONE = auto()
    BASIC = auto()
    ROTATION = auto()
    FRICTION = auto()

    @property
    def is_contact_mode(self) -> bool:
        """True for the modes that cache per-contact data and the batched kernel solves."""
        return self in (PhysicsMode.ROTATION, PhysicsMode.FRICTION)


class DetectionKind(Enum):
    """The broad-phase collision-detection algorithm to use."""

    BASIC = 1
    SPATIAL_HASHING = 2
    QUADTREE = 3
    LOOSE_QUADTREE = 4
