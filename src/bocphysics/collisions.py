"""Module providing routines for collision detection."""

import math
from typing import NamedTuple

from bocpy import Matrix

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


def batched_circle_circle(pairs):
    """Resolve K circle-circle pairs at once; return one Collision-or-None per pair."""
    k = len(pairs)
    ax = Matrix(k, 1, [a.position.x for a, b in pairs])
    ay = Matrix(k, 1, [a.position.y for a, b in pairs])
    bx = Matrix(k, 1, [b.position.x for a, b in pairs])
    by = Matrix(k, 1, [b.position.y for a, b in pairs])
    rsum = Matrix(k, 1, [(a.radius + b.radius) * 0.97 for a, b in pairs])
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    length = length2.sqrt()
    depth = rsum - length
    nx = dx.divide(length)
    ny = dy.divide(length)
    out = []
    for i, (_a, _b) in enumerate(pairs):
        if length2[i, 0] >= rsum[i, 0] ** 2:
            out.append(None)
            continue
        out.append(Collision(Matrix.vector([nx[i, 0], ny[i, 0]]), depth[i, 0]))
    return out


def batched_circle_polygon(pairs):
    """Resolve K (circle, polygon) pairs at once; return one Collision-or-None per pair."""
    k = len(pairs)
    acap = max(p.transformed_normals.rows for _, p in pairs)
    cap = acap + 1
    pvx, pvy, nxs, nys, valid = [], [], [], [], []
    cx, cy, rad, dpx, dpy = [], [], [], [], []
    for c, p in pairs:
        pv = p.transformed_vertices
        pn = p.transformed_normals
        n = pv.rows
        kn = pn.rows
        vx = [pv[i, 0] for i in range(n)] + [pv[0, 0]] * (_MAX_VERTS - n)
        vy = [pv[i, 1] for i in range(n)] + [pv[0, 1]] * (_MAX_VERTS - n)
        pvx.append(vx)
        pvy.append(vy)
        nxs.append([pn[i, 0] for i in range(kn)] + [0.0] * (acap - kn))
        nys.append([pn[i, 1] for i in range(kn)] + [0.0] * (acap - kn))
        valid.append([1.0] * kn + [0.0] * (acap - kn))
        cx.append([c.position.x])
        cy.append([c.position.y])
        rad.append([c.radius * 0.97])
        dpx.append([p.position.x - c.position.x])
        dpy.append([p.position.y - c.position.y])

    def block(rows, w):
        return Matrix(k, w, [v for row in rows for v in row])

    px, py = block(pvx, _MAX_VERTS), block(pvy, _MAX_VERTS)
    cxm, cym, radm = block(cx, 1), block(cy, 1), block(rad, 1)
    # closest poly vertex to the circle center; padded slots duplicate vertex 0 so never win
    d2 = None
    for v in range(_MAX_VERTS):
        dx = px[:, v] - cxm
        dy = py[:, v] - cym
        col = dx * dx + dy * dy
        d2 = col if d2 is None else Matrix.concat([d2, col], 1)
    hot = Matrix.equal(Matrix(1, _MAX_VERTS, [float(j) for j in range(_MAX_VERTS)]),
                       d2.argmin(axis=1))
    dfx = hot.multiply(px).sum(axis=1) - cxm
    dfy = hot.multiply(py).sum(axis=1) - cym
    length = (dfx * dfx + dfy * dfy).sqrt()
    nx = Matrix.concat([block(nxs, acap), dfx.divide(length)], 1)
    ny = Matrix.concat([block(nys, acap), dfy.divide(length)], 1)
    pmin = pmax = None
    for v in range(_MAX_VERTS):
        proj = px[:, v] * nx + py[:, v] * ny
        if v == 0:
            pmin = pmax = proj
        else:
            pmin = Matrix.where(Matrix.less(proj, pmin), proj, pmin)
            pmax = Matrix.where(Matrix.less(pmax, proj), proj, pmax)
    center = cxm * nx + cym * ny
    cmin, cmax = center - radm, center + radm
    separated = (Matrix.less(cmax, pmin) + Matrix.less(pmax, cmin)).max(axis=1)
    depth = Matrix.where(Matrix.less(cmax - pmin, pmax - cmin), cmax - pmin, pmax - cmin)
    mask = Matrix.concat([block(valid, acap), Matrix(k, 1, [1.0] * k)], 1)
    depth = Matrix.where(mask, depth, Matrix(k, cap, [_BIG] * (k * cap)))
    chosen = Matrix.equal(Matrix(1, cap, [float(j) for j in range(cap)]), depth.argmin(axis=1))
    nsx = chosen.multiply(nx).sum(axis=1)
    nsy = chosen.multiply(ny).sum(axis=1)
    depth_min = depth.min(axis=1)
    dot = block(dpx, 1) * nsx + block(dpy, 1) * nsy
    sign = Matrix.where(Matrix.less(dot, Matrix(k, 1, [0.0] * k)),
                        Matrix(k, 1, [-1.0] * k), Matrix(k, 1, [1.0] * k))
    out = []
    for i in range(k):
        if separated[i, 0] > 0:
            out.append(None)
            continue
        s = sign[i, 0]
        out.append(Collision(Matrix.vector([nsx[i, 0] * s, nsy[i, 0] * s]), depth_min[i, 0]))
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
