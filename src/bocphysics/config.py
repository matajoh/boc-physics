"""Configuration data structures for the physics engine.

Collects the typed, picklable settings the engine and renderer read --
window :class:`Resolution`, the :class:`DetectionKind` broad-phase choice,
and the tunable constants.
"""

from enum import Enum
from typing import NamedTuple


class Resolution(NamedTuple("Resolution", [("width", int), ("height", int)])):
    """Resolution of the simulation window."""

    @staticmethod
    def from_string(name: str) -> "Resolution":
        """Parse a resolution from a "WIDTHxHEIGHT" string."""
        parts = name.split("x")
        return Resolution(int(parts[0]), int(parts[1]))


class DetectionKind(Enum):
    """The broad-phase collision-detection algorithm to use."""

    # Selected by name (DetectionKind[...]); the integer values are never persisted or shared.
    BASIC = 1
    QUADTREE = 2
    LOOSE_QUADTREE = 3
