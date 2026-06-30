"""Module providing routines for collision detection."""

import math
from typing import NamedTuple

from bocpy import Matrix

from . import transport
from .bodies import Circle, Polygon, RigidBody

_MAX_VERTS = 8          # widest polygon in the scene; pads the batched vertex stack
_BIG = 1.0e30           # sentinel depth for masked padding axes


class Collision(NamedTuple("Collision", [("normal", Matrix), ("depth", float)])):
    """A collision described by its contact normal and penetration depth."""

    def reverse(self) -> "Collision":
        """Return a collision with the normal pointing the other way."""
        return Collision(-self.normal, self.depth)


def intersect_circle_circle(a: Circle, b: Circle) -> Collision:
    """Determine if two circles intersect and return the collision."""
    diff = b.position - a.position
    diff_length2 = diff.vecdot(diff)
    # shrink the hitbox ~3% so drawn circles appear to touch at contact
    radius_sum = (a.radius + b.radius) * 0.97
    if diff_length2 >= radius_sum**2:
        return None

    diff_length = math.sqrt(diff_length2)
    normal = diff / diff_length
    depth = radius_sum - diff_length
    return Collision(normal, depth)


def closest_vertex_on_polygon(point: Matrix, poly: Polygon) -> Matrix:
    """Find the closest vertex on a polygon to a point."""
    verts = poly.transformed_vertices
    return verts[(verts - point).magnitude_squared(axis=1).argmin()]


def intersect_circle_polygon(circle: Circle, poly: Polygon) -> Collision:
    """Determine if a circle and a polygon intersect and return the collision."""
    closest_point = closest_vertex_on_polygon(circle.position, poly)
    diff = closest_point - circle.position
    normals = Matrix.concat([poly.transformed_normals, diff.normalize()], 0)
    poly_proj = poly.transformed_vertices @ normals.T
    rect_min, rect_max = poly_proj.min(axis=0), poly_proj.max(axis=0)
    center = circle.position @ normals.T
    # shrink the hitbox ~3% so drawn circles appear to touch at contact
    radius = circle.radius * 0.97
    circ_min, circ_max = center - radius, center + radius
    if (Matrix.less(circ_max, rect_min) + Matrix.less(rect_max, circ_min)).max() > 0:
        return None

    pen1 = circ_max - rect_min
    pen2 = rect_max - circ_min
    depth = Matrix.where(Matrix.less(pen1, pen2), pen1, pen2)
    axis = normals[depth.argmin()]
    if (poly.position - circle.position).vecdot(axis) < 0:
        axis = -axis

    return Collision(axis, depth.min())


def intersect_polygon_polygon(a: Polygon, b: Polygon) -> Collision:
    """Determine if two polygons intersect and return the collision."""
    normals = Matrix.concat([a.transformed_normals, b.transformed_normals], 0)
    nt = normals.T
    a_proj = a.transformed_vertices @ nt
    b_proj = b.transformed_vertices @ nt
    a_min, a_max = a_proj.min(axis=0), a_proj.max(axis=0)
    b_min, b_max = b_proj.min(axis=0), b_proj.max(axis=0)
    if (Matrix.less(a_max, b_min) + Matrix.less(b_max, a_min)).max() > 0:
        return None

    pen1 = a_max - b_min
    pen2 = b_max - a_min
    depth = Matrix.where(Matrix.less(pen1, pen2), pen1, pen2)
    axis = normals[depth.argmin()]
    if (b.position - a.position).vecdot(axis) < 0:
        axis = -axis

    return Collision(axis, depth.min())


def _circle_center(circle, state):
    """Circle centre (x, y): dynamic from the State block by uid, static from the body."""
    if state is not None and circle.physics:
        row = state.row_of.get(circle.uid)
        if row is not None:
            return (state.block[row, transport.POSITION.start],
                    state.block[row, transport.POSITION.start + 1])
    return circle.position.x, circle.position.y


def batched_circle_circle(pairs, state=None):
    """Resolve K circle-circle pairs at once; return one Collision-or-None per pair."""
    k = len(pairs)
    dx, dy, rsum = [0] * k, [0] * k, [0] * k
    for i, (a, b) in enumerate(pairs):
        ax, ay = _circle_center(a, state)
        bx, by = _circle_center(b, state)
        dx[i] = bx - ax
        dy[i] = by - ay
        rsum[i] = (a.radius + b.radius) * 0.97
    dx = Matrix(k, 1, dx)
    dy = Matrix(k, 1, dy)
    rsum = Matrix(k, 1, rsum)
    length2 = dx * dx
    length2.scaled_add(dy, dy, in_place=True)
    length = length2.sqrt()
    depth = rsum - length
    nx = dx / length
    ny = dy / length
    rsum *= rsum
    disjoint = length2.greater_equal(rsum)
    out = [None] * k
    for i, (is_disjoint, x, y, d) in enumerate(zip(disjoint, nx, ny, depth)):
        if is_disjoint:
            continue

        out[i] = Collision(Matrix.vector([x, y]), d)
    return out


def batched_circle_polygon(pairs, geom, state=None):
    """Resolve K (circle, polygon) pairs at once; return one Collision-or-None per pair.

    geom is the patch GeometryPool: poly verts/normals are read as whole rows by
    uid via take, padded to the pool's vmax/nmax (vertex 0 / zero normals), so no
    per-pair unbox. acap is the pool-wide normal count; spare slots never win.
    """
    k = len(pairs)
    acap = geom.nmax
    cap = acap + 1
    rows = [geom.row_of[p.uid] for _, p in pairs]
    cx = [0] * k
    cy = [0] * k
    rad = [0] * k
    dpx = [0] * k
    dpy = [0] * k
    valid = Matrix.zeros((k, acap))
    for i, (c, p) in enumerate(pairs):
        cx[i], cy[i] = _circle_center(c, state)
        rad[i] = c.radius * 0.97
        dpx[i] = p.position.x - cx[i]
        dpy[i] = p.position.y - cy[i]
        valid[i, :len(p.normals)] = 1.0

    vmax = geom.vmax
    pvx, pvy = geom.geom_x.take(rows, 0), geom.geom_y.take(rows, 0)
    nxs, nys = geom.norm_x.take(rows, 0), geom.norm_y.take(rows, 0)
    cxm, cym, radm = Matrix(k, 1, cx), Matrix(k, 1, cy), Matrix(k, 1, rad)
    # closest poly vertex to the circle center; padded slots duplicate vertex 0 so never win
    dx = pvx - cxm
    dy = pvy - cym
    d2 = dx * dx
    d2.scaled_add(dy, dy, in_place=True)
    nearest = d2.argmin(axis=1)
    dfx = dx.take_along_axis(nearest, axis=1)
    dfy = dy.take_along_axis(nearest, axis=1)
    length = d2.take_along_axis(nearest, axis=1)
    length.sqrt(in_place=True)
    nx = Matrix.concat([nxs, dfx.divide(length)], 1)
    ny = Matrix.concat([nys, dfy.divide(length)], 1)
    pmin = Matrix.full((k, cap), _BIG)
    pmax = Matrix.full((k, cap), -_BIG)
    proj = Matrix.zeros((k, cap))
    for v in range(vmax):
        Matrix.multiply(nx, pvx[:, v], out=proj)
        proj.scaled_add(pvy[:, v], ny, in_place=True)
        Matrix.where(Matrix.less(proj, pmin), proj, pmin, out=pmin)
        Matrix.where(Matrix.less(pmax, proj), proj, pmax, out=pmax)

    center = cxm * nx
    center.scaled_add(cym, ny, in_place=True)
    cmin, cmax = center - radm, center + radm
    Matrix.subtract(cmax, pmin, out=pmin)
    Matrix.subtract(pmax, cmin, out=pmax)
    depth = Matrix.where(Matrix.less(pmin, pmax), pmin, pmax)
    mask = Matrix.concat([valid, Matrix.ones((k, 1))], 1)
    depth = Matrix.where(mask, depth, _BIG)
    chosen = depth.argmin(axis=1)
    nsx = nx.take_along_axis(chosen, axis=1)
    nsy = ny.take_along_axis(chosen, axis=1)
    depth_min = depth.take_along_axis(chosen, axis=1)
    sign = Matrix(k, 1, dpx) * nsx + Matrix(k, 1, dpy) * nsy
    Matrix.where(Matrix.less(sign, 0), -1, 1, out=sign)
    disjoint = (Matrix.less(pmin, 0) + Matrix.less(pmax, 0)).max(axis=1)
    Matrix.greater(disjoint, 0, out=disjoint)
    out = [None] * k
    for i, (is_disjoint, x, y, s, d) in enumerate(zip(disjoint, nsx, nsy, sign, depth_min)):
        if is_disjoint:
            continue

        out[i] = Collision(Matrix.vector([x * s, y * s]), d)
    return out


def batched_polygon_polygon(pairs, geom):
    """Resolve K (polygon, polygon) pairs at once; return one Collision-or-None per pair.

    geom is the patch GeometryPool. SAT axes are both polys' normals padded to
    nmax each (zero normals masked off); verts read as whole rows by uid via take,
    padded with vertex 0 so spare slots never extend a projection interval.
    """
    k = len(pairs)
    ncap = 2 * geom.nmax
    vmax = geom.vmax
    rows_a = [geom.row_of[a.uid] for a, _ in pairs]
    rows_b = [geom.row_of[b.uid] for _, b in pairs]
    valid = Matrix.zeros((k, ncap))
    dpx = [0] * k
    dpy = [0] * k
    for i, (a, b) in enumerate(pairs):
        valid[i, :len(a.normals)] = 1.0
        valid[i, geom.nmax:geom.nmax + len(b.normals)] = 1.0
        dpx[i] = b.position.x - a.position.x
        dpy[i] = b.position.y - a.position.y
    avx, avy = geom.geom_x.take(rows_a, 0), geom.geom_y.take(rows_a, 0)
    bvx, bvy = geom.geom_x.take(rows_b, 0), geom.geom_y.take(rows_b, 0)
    nx = Matrix.concat([geom.norm_x.take(rows_a, 0), geom.norm_x.take(rows_b, 0)], 1)
    ny = Matrix.concat([geom.norm_y.take(rows_a, 0), geom.norm_y.take(rows_b, 0)], 1)
    amin = Matrix.full((k, ncap), _BIG)
    bmin = Matrix.full((k, ncap), _BIG)
    amax = Matrix.full((k, ncap), -_BIG)
    bmax = Matrix.full((k, ncap), -_BIG)
    proj = Matrix.zeros((k, ncap))
    for v in range(vmax):
        Matrix.multiply(nx, avx[:, v], out=proj)
        proj.scaled_add(avy[:, v], ny, in_place=True)
        Matrix.where(Matrix.less(proj, amin), proj, amin, out=amin)
        Matrix.where(Matrix.less(amax, proj), proj, amax, out=amax)

        Matrix.multiply(nx, bvx[:, v], out=proj)
        proj.scaled_add(bvy[:, v], ny, in_place=True)
        Matrix.where(Matrix.less(proj, bmin), proj, bmin, out=bmin)
        Matrix.where(Matrix.less(bmax, proj), proj, bmax, out=bmax)

    pen1, pen2 = amax - bmin, bmax - amin
    depth = Matrix.where(Matrix.less(pen1, pen2), pen1, pen2)
    depth = Matrix.where(valid, depth, _BIG)
    chosen = depth.argmin(axis=1)
    nsx = nx.take_along_axis(chosen, axis=1)
    nsy = ny.take_along_axis(chosen, axis=1)
    depth_min = depth.take_along_axis(chosen, axis=1)
    dot = Matrix(k, 1, dpx) * nsx
    dot.scaled_add(Matrix(k, 1, dpy), nsy, in_place=True)
    sign = Matrix.where(Matrix.less(dot, 0), -1, 1)
    disjoint = (Matrix.less(pen1, 0) + Matrix.less(pen2, 0)).max(axis=1)
    Matrix.greater(disjoint, 0, out=disjoint)
    out = [None] * k
    for i, (is_disjoint, x, y, s, d) in enumerate(zip(disjoint, nsx, nsy, sign, depth_min)):
        if is_disjoint:
            continue

        out[i] = Collision(Matrix.vector([x * s, y * s]), d)

    return out


def detect_collision(a: RigidBody, b: RigidBody) -> Collision:
    """Dispatch to the right narrow-phase test for the body pair."""
    if isinstance(a, Circle):
        if isinstance(b, Circle):
            return intersect_circle_circle(a, b)

        return intersect_circle_polygon(a, b)
    elif isinstance(a, Polygon):
        if isinstance(b, Circle):
            collision = intersect_circle_polygon(b, a)
            return collision.reverse() if collision else None

        return intersect_polygon_polygon(a, b)
