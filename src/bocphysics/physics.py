"""Module providing the physics system."""

from typing import NamedTuple


class Physics(NamedTuple):
    """Immutable physics-config snapshot shared across worker sub-interpreters.

    Description:
        A body's ``.physics`` flag -- not this type -- decides which bodies the
        solver integrates and pushes; this snapshot only carries the friction and
        restitution coefficients the XPBD solve reads. It is an immutable
        NamedTuple of plain values so it can ride on the noticeboard and be read
        by every worker sub-interpreter as a shared config snapshot.
    """

    restitution: float = 0.5
    static_friction: float = 0.5
    dynamic_friction: float = 0.5
