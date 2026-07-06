"""Module providing the physics system."""

from typing import NamedTuple


class Physics(NamedTuple):
    """Immutable physics-config snapshot for the contact solve.

    Description:
        A body's ``.physics`` flag -- not this type -- decides which bodies the
        solver integrates and pushes; this snapshot only carries the friction and
        restitution coefficients the XPBD solve reads. It is an immutable
        NamedTuple of plain values shared by every sub-step of a frame.
    """

    restitution: float = 0.5
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
