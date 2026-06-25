"""Configuration data structures for the physics engine.

Collects the typed, picklable settings the engine and renderer read --
window :class:`Resolution`, the :class:`PhysicsMode` collision model, the
:class:`DetectionKind` broad-phase choice, and the tunable constants.
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

    # Selected by name (DetectionKind[...]); the integer values are never persisted or shared.
    BASIC = 1
    QUADTREE = 2
    LOOSE_QUADTREE = 3
