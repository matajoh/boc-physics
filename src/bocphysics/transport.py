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

import math
from typing import List, Optional, Tuple

from bocpy import Matrix

from .bodies import Polygon, RigidBody

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


def assert_block_mirrors(block: Matrix, row_of: dict, bodies: List[RigidBody]):
    """Bridge safety net: assert every dynamic body's block row equals its scalar pose.

    Description:
        The B-bridge keeps the scalar bodies as a write-through mirror of the
        State block while readers move onto the block one at a time. This guard
        catches a column-layout or row-order mismatch the instant a reader
        starts trusting the block. The block stores body.position.x/.y and
        body.angle as exact float64 copies, so the comparison is exact. Bodies
        with no row (statics) are skipped; the mirror is removed at B6.
    """
    for body in bodies:
        row = row_of.get(body.uid)
        if row is None:
            continue
        assert block[row, POSITION.start] == body.position.x
        assert block[row, POSITION.start + 1] == body.position.y
        assert block[row, ANGLE] == body.angle


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


class State:
    """Authoritative (N x 7) pool for one body set; bodies become views into it.

    Description:
        Owns one persistent block and a uid->row map rebuilt only when the body
        set changes. gather seeds the pool from scalar bodies; scatter writes it
        back. The block is the same column layout pack_state uses, so it crosses
        a cown by pointer. Statics are excluded -- they never integrate.
    """

    def __init__(self, bodies: List[RigidBody]):
        """Build the pool and row map from the dynamic bodies in order."""
        self.rebuild(bodies)

    def rebuild(self, bodies: List[RigidBody]):
        """Reseed pool and row_of after a body-set change; statics excluded."""
        self.bodies = [b for b in bodies if b.physics]
        self.row_of = {b.uid: i for i, b in enumerate(self.bodies)}
        self.block = pack_state(self.bodies) if self.bodies else None

    def gather(self):
        """Write every body's mutable state into the pool block."""
        if self.block is None:
            return
        store_state(self.bodies, self.block)

    def scatter(self):
        """Read the pool block back onto the bodies."""
        if self.block is None:
            return
        apply_state(self.bodies, self.block)


class GeometryPool:
    """Patch-wide transformed polygon geometry as padded SoA rows.

    Description:
        One row per polygon, statics included; circles have no geometry and are
        absent. Local vertices/normals are stored padded once at rebuild; sync
        rotates+translates the whole base block in place -- one batched column
        rotation, no per-poly Python loop -- so geom_x/geom_y/norm_x/norm_y hold
        the current world pose. The contact batchers read rows by uid via take.
    """

    def __init__(self, bodies: List[RigidBody]):
        """Build the geometry rows and uid->row map from the polygons."""
        self.rebuild(bodies)

    def rebuild(self, bodies: List[RigidBody]):
        """Reseed padded base rows and row_of after a body-set change; sync once."""
        self.polys = [b for b in bodies if isinstance(b, Polygon)]
        self.row_of = {b.uid: i for i, b in enumerate(self.polys)}
        self.vcount = {}
        if not self.polys:
            self.vmax = self.nmax = 0
            self.geom_x = self.geom_y = self.norm_x = self.norm_y = None
            return

        vmax = max(len(p.vertices) for p in self.polys)
        nmax = max(len(p.normals) for p in self.polys)
        self.vmax = vmax
        self.nmax = nmax
        rows = len(self.polys)
        vx = [0] * rows * vmax
        vy = [0] * rows * vmax
        nx = [0] * rows * nmax
        ny = [0] * rows * nmax
        vscan = 0
        nscan = 0
        for p in self.polys:
            self.vcount[p.uid] = len(p.vertices)
            for i, v in enumerate(p.vertices):
                vx[vscan + i] = v.x
                vy[vscan + i] = v.y
            for i in range(len(p.vertices), vmax):
                vx[vscan + i] = p.vertices[0].x
                vy[vscan + i] = p.vertices[0].y

            for i, n in enumerate(p.normals):
                nx[nscan + i] = n.x
                ny[nscan + i] = n.y

            vscan += vmax
            nscan += nmax

        self.base_vx = Matrix(rows, vmax, vx)
        self.base_vy = Matrix(rows, vmax, vy)
        self.base_nx = Matrix(rows, nmax, nx)
        self.base_ny = Matrix(rows, nmax, ny)
        self.px = [0] * rows
        self.py = [0] * rows
        self.cos = [0] * rows
        self.sin = [0] * rows
        self.sync()

    def sync(self):
        """Refresh world pose from the polygons' current scalar transforms."""
        for i, p in enumerate(self.polys):
            self.cos[i] = math.cos(p.angle)
            self.sin[i] = math.sin(p.angle)
            self.px[i] = p.position.x
            self.py[i] = p.position.y
        self._apply_pose()

    def sync_from(self, px, py, angle):
        """Refresh world pose from packed pose columns instead of scalar bodies.

        Description:
            px, py, and angle are per-row sequences in self.polys order. cos and
            sin are taken per element with math so the rotation is bit-for-bit
            identical to the body-sourced sync; only the pose SOURCE differs.
        """
        for i in range(len(self.polys)):
            self.cos[i] = math.cos(angle[i])
            self.sin[i] = math.sin(angle[i])
            self.px[i] = px[i]
            self.py[i] = py[i]
        self._apply_pose()

    def sync_from_block(self, block: Matrix, row_of: dict):
        """Refresh world pose from a packed dynamics block; statics keep rebuild pose.

        Description:
            row_of maps a dynamic body's uid to its row in the (N x 7) block. Each
            dynamic polygon reads its pose from that row with no scalar body
            access; static polygons are absent from the block and retain the pose
            captured at rebuild in self.px/py/cos/sin. cos/sin are taken per
            element with math so the result matches the body-sourced sync exactly.
        """
        for i, p in enumerate(self.polys):
            r = row_of.get(p.uid)
            if r is None:
                continue
            self.px[i] = block[r, POSITION.start]
            self.py[i] = block[r, POSITION.start + 1]
            angle = block[r, ANGLE]
            self.cos[i] = math.cos(angle)
            self.sin[i] = math.sin(angle)
        self._apply_pose()

    def _apply_pose(self):
        """Rotate+translate the base block into world pose, one batched pass."""
        rows = len(self.polys)
        if rows == 0:
            return
        cos = Matrix(rows, 1, self.cos)
        sin = Matrix(rows, 1, self.sin)
        px = Matrix(rows, 1, self.px)
        py = Matrix(rows, 1, self.py)

        self.geom_x = self.base_vx * cos
        self.geom_x -= self.base_vy * sin
        self.geom_x += px

        self.geom_y = self.base_vx * sin
        self.geom_y += self.base_vy * cos
        self.geom_y += py

        self.norm_x = self.base_nx * cos
        self.norm_x -= self.base_ny * sin

        self.norm_y = self.base_nx * sin
        self.norm_y += self.base_ny * cos

    def world_vertices(self, uid) -> Matrix:
        """Return one polygon's transformed vertices as a real-count (nv x 2) block."""
        r = self.row_of[uid]
        nv = self.vcount[uid]
        return Matrix.concat([self.geom_x[r, :nv].T, self.geom_y[r, :nv].T], 1)
