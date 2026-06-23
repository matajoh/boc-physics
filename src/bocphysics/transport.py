"""Patch mutable-state transport between the main interpreter and BOC workers.

Description:
    A patch's per-substep mutable state is one contiguous (N x 7) Matrix block:
    each row is a body, the columns are uid, position (x, y), velocity (x, y),
    angle, and spin. A bare Matrix is the only thing XIData hands across a
    sub-interpreter by pointer -- measured flat ~0.2us regardless of N -- whereas
    a tuple, NamedTuple, or even a plain int list pickles every member by value
    and scales with N. So the whole patch travels as ONE Matrix; the immutable
    geometry rides the noticeboard separately. The named column slices below keep
    the packed layout readable; never reorder them without updating WIDTH.
"""

from typing import List, Optional, Tuple

from bocpy import Matrix

from .bodies import RigidBody

UID = 0
POSITION = slice(1, 3)
VELOCITY = slice(3, 5)
ANGLE = 5
SPIN = 6
WIDTH = 7


def pack_state(bodies: List[RigidBody]) -> Matrix:
    """Gather the dynamic bodies' uid and mutable state into one (N x 7) block."""
    n = len(bodies)
    data = []
    for body in bodies:
        data.extend((float(body.uid), body.position.x, body.position.y,
                     body.linear_velocity.x, body.linear_velocity.y,
                     body.angle, body.angular_velocity))
    return Matrix(n, WIDTH, data)


def apply_state(bodies: List[RigidBody], block: Matrix):
    """Scatter a packed (N x 7) block back onto the matching bodies by row order.

    Description:
        Bodies must be in the same order pack_state saw them so the rows line up
        with the uid column. Sets the dirty bit so world geometry is lazily
        recomputed at the new pose.
    """
    position = block[:, POSITION]
    velocity = block[:, VELOCITY]
    for i, body in enumerate(bodies):
        body.position = position[i]
        body.linear_velocity = velocity[i]
        body.angle = block[i, ANGLE]
        body.angular_velocity = block[i, SPIN]
        body.update_needed_ = True


def store_state(bodies: List[RigidBody], block: Matrix):
    """Write the bodies' mutable state back into an existing block in place.

    Description:
        The inverse of apply_state, but it overwrites the block the cown already
        owns instead of allocating a fresh (N x 7) Matrix every sub-step. The two
        2-wide columns go back as whole-slice Matrix assignments (one fast C copy
        each); the scalar columns are set per row. The uid column is untouched.
    """
    n = len(bodies)
    pos_data = []
    vel_data = []
    for body in bodies:
        pos_data.extend((body.position.x, body.position.y))
        vel_data.extend((body.linear_velocity.x, body.linear_velocity.y))
    block[:, POSITION] = Matrix(n, 2, pos_data)
    block[:, VELOCITY] = Matrix(n, 2, vel_data)
    for i, body in enumerate(bodies):
        block[i, ANGLE] = body.angle
        block[i, SPIN] = body.angular_velocity


def uids_of(block: Matrix) -> List[int]:
    """Read the uid column of a packed block back as a list of ints."""
    return [int(block[i, UID]) for i in range(block.rows)]


def pack_pairs(pairs: List[Tuple[int, int]]) -> Optional[Matrix]:
    """Pack interior uid pairs into an (M x 2) block, or None when there are none.

    Description:
        A zero-row Matrix is not constructible, so an empty pair list packs to
        None. This block is reused across every sub-step behavior, so it rides in
        its own cown rather than a closure capture, which would be moved away.
    """
    if not pairs:
        return None
    data = [float(uid) for pair in pairs for uid in pair]
    return Matrix(len(pairs), 2, data)


def unpack_pairs(block: Optional[Matrix]) -> List[Tuple[int, int]]:
    """Read an (M x 2) pair block back as a list of (uid, uid) tuples."""
    if block is None:
        return []
    return [(int(block[i, 0]), int(block[i, 1])) for i in range(block.rows)]
