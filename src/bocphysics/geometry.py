"""Shared polygon geometry: contact-point slip, broad-phase boxes, and the batched geometry pool."""

import math
from typing import Optional

from bocpy import Matrix

from .bodies import AABB, Polygon, RigidBody


def rotate(v: Matrix, angle: Matrix) -> Matrix:
    """Rotate a 2D vector CCW by angle (radians) via the perpendicular basis.

    Description:
        R(a) v = v cos a + perp(v) sin a, so this needs only scalar-multiply,
        add, and perpendicular -- the ops a contact lever arm supports whether it
        is a Matrix.vector (circle) or a manifold row slice (polygon). It never
        touches v.x / v.y, which a slice does not expose.
    """
    return v * angle.cos() + v.perpendicular() * angle.sin()


ZEROVEC2 = Matrix.zeros((1, 2))


def contact_point_slip(body: RigidBody, r: Matrix, prev: Optional[Matrix]) -> Matrix:
    """World displacement of the contact's material point on body over the sub-step.

    Description:
        The point rides rigidly with the body, so its previous world position is
        the previous centre plus the lever arm rotated back by the pose change. A
        static body (absent from prev_pose) never moves, so its point does not
        slip. Returns current minus previous position, in world space.
    """
    if prev is None:
        return ZEROVEC2

    return (body.position + r) - (prev[0, :2] + rotate(r, prev.z - body.angle))


def broad_box(body: RigidBody) -> AABB:
    """Conservative bounding-circle AABB from the body's scalar pose.

    Description:
        The broad-phase cull only needs a box that never shrinks below the true
        bounds, so the rotation-invariant bounding circle (centre +/- radius)
        suffices and needs no angle. The looser box only widens the cull -- it
        never rejects a real overlap -- so the emitted constraint set and its
        order are unchanged.
    """
    x, y = body.position.x, body.position.y
    rad = body.radius
    return AABB(x - rad, y - rad, x + rad, y + rad)


class GeometryPool:
    """Transformed polygon geometry as padded SoA rows for batched SAT.

    Description:
        One row per polygon, statics included; circles have no geometry and are
        absent. Local vertices/normals are stored padded once at rebuild; sync
        rotates+translates the whole base block in place -- one batched column
        rotation, no per-poly Python loop -- so geom_x/geom_y/norm_x/norm_y hold
        the current world pose. The contact batchers read rows by uid via take.
    """

    def __init__(self, bodies: list[RigidBody]):
        """Build the geometry rows and uid->row map from the polygons."""
        self.rebuild(bodies)

    def rebuild(self, bodies: list[RigidBody]):
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
            self.cos[i] = p.angle.cos()
            self.sin[i] = p.angle.sin()
            self.px[i] = p.position.x
            self.py[i] = p.position.y
        self._apply_pose()

    def _apply_pose(self):
        """Rotate+translate the base block into world pose, one batched pass."""
        rows = len(self.polys)
        if rows == 0:
            return
        cos = Matrix.concat(self.cos)
        sin = Matrix.concat(self.sin)
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
