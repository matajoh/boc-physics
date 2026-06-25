"""Module providing rendering helpers shared by the bodies and engine.

This module isolates the pyglet rendering details (colour conversion and
the world-to-screen projection) behind a small seam so the physics code
never imports pyglet directly.
"""

import math
from typing import Tuple, Union

from bocpy import Matrix
import webcolors

from .bodies import Circle


Color = Union[str, Tuple[int, int, int], Tuple[int, int, int, int]]
RGBA = Tuple[int, int, int, int]

BLACK = (0, 0, 0, 255)
YELLOW = (255, 255, 0, 255)
OVERLAY_ORDER = 1_000_000
SLAB_COLOR = (200, 40, 40, 200)
QUADTREE_COLOR = (40, 100, 200, 200)


def to_rgba(color: Color) -> RGBA:
    """Convert a colour name or RGB(A) tuple to an RGBA tuple for pyglet."""
    if isinstance(color, str):
        r, g, b = webcolors.name_to_rgb(color)
        return (r, g, b, 255)

    if len(color) == 3:
        return (color[0], color[1], color[2], 255)

    return (color[0], color[1], color[2], color[3])


class Camera:
    """Projects world coordinates (metres) to pyglet screen pixels.

    Description:
        The physics world is y-down with the origin at the top-left;
        pyglet is y-up with the origin at the bottom-left. The flip is
        applied here, at the projection boundary, so the physics math is
        never affected.
    """

    def __init__(self, center: Matrix, scale: float, height: float):
        """Create a camera from a world centre offset, pixel scale, and height."""
        self.center = center
        self.scale = scale
        self.height = height

    def __call__(self, point: Matrix) -> Tuple[float, float]:
        """Project a world point to a screen (x, y) pixel coordinate."""
        x = (point.x + self.center.x) * self.scale
        y = (point.y + self.center.y) * self.scale
        return x, self.height - y


def draw_body(body, batch, project: Camera, fill_group, line_group) -> Tuple:
    """Draw one body into the batch by type, returning the shapes to keep alive."""
    from pyglet import shapes
    if isinstance(body, Circle):
        x, y = project(body.position)
        radius = body.radius * project.scale
        p = Matrix.vector([math.cos(body.angle), math.sin(body.angle)]) * radius * 0.9
        fill = shapes.Circle(x, y, radius, color=to_rgba(body.color), batch=batch, group=fill_group)
        if not body.physics:
            outline = shapes.Arc(x, y, radius, closed=True, thickness=4, color=BLACK, batch=batch, group=line_group)
            return fill, outline

        outline = shapes.Arc(x, y, radius, thickness=4, color=BLACK, batch=batch, group=line_group)
        heading = shapes.Line(x, y, x + p.x, y - p.y, thickness=4, color=BLACK, batch=batch, group=line_group)
        return fill, outline, heading

    vertices = [project(v) for v in body.transformed_vertices]
    fill = shapes.Polygon(*vertices, color=to_rgba(body.color), batch=batch, group=fill_group)
    outline = shapes.MultiLine(*vertices, closed=True, thickness=4, color=BLACK, batch=batch, group=line_group)
    return (fill, outline)


def draw_static_layer(bodies, batch, project: Camera) -> list:
    """Draw stationary bodies with per-body ordered groups so overlaps layer correctly."""
    from pyglet import graphics
    kept = []
    order = 0
    for body in bodies:
        fill_group = graphics.Group(order=order)
        line_group = graphics.Group(order=order + 1)
        order += 2
        kept.extend(draw_body(body, batch, project, fill_group, line_group))

    return kept


def draw_frame(bodies, contacts, batch, project: Camera) -> list:
    """Draw the moving bodies and contact points; return the shapes to keep alive."""
    from pyglet import graphics, shapes
    fill_group = graphics.Group(order=0)
    line_group = graphics.Group(order=1)
    mark_group = graphics.Group(order=2)
    kept = []
    for body in bodies:
        kept.extend(draw_body(body, batch, project, fill_group, line_group))

    for contact in contacts:
        x, y = project(Matrix.vector([contact[0], contact[1]]))
        kept.append(shapes.Circle(x, y, 5, color=YELLOW, batch=batch, group=mark_group))
        kept.append(shapes.Arc(x, y, 5, thickness=2, color=BLACK, batch=batch, group=mark_group))

    return kept


def draw_slab_overlay(boundaries, top: float, bottom: float, batch, project: Camera) -> list:
    """Draw a vertical line at each slab seam x; return the shapes to keep alive."""
    from pyglet import graphics, shapes
    group = graphics.Group(order=OVERLAY_ORDER)
    kept = []
    for x in boundaries:
        x0, y0 = project(Matrix.vector([x, top]))
        x1, y1 = project(Matrix.vector([x, bottom]))
        kept.append(shapes.Line(x0, y0, x1, y1, thickness=2, color=SLAB_COLOR, batch=batch, group=group))

    return kept


def draw_box_overlay(boxes, batch, project: Camera) -> list:
    """Outline each AABB (e.g. quadtree cells); return the shapes to keep alive."""
    from pyglet import graphics, shapes
    group = graphics.Group(order=OVERLAY_ORDER)
    kept = []
    for box in boxes:
        corners = [project(Matrix.vector([box.left, box.top])),
                   project(Matrix.vector([box.right, box.top])),
                   project(Matrix.vector([box.right, box.bottom])),
                   project(Matrix.vector([box.left, box.bottom]))]
        kept.append(shapes.MultiLine(*corners, closed=True, thickness=1,
                                     color=QUADTREE_COLOR, batch=batch, group=group))

    return kept


def open_encoder(path: str, width: int, height: int, fps: int):
    """Start an ffmpeg subprocess that consumes raw RGBA frames over stdin."""
    import subprocess

    try:
        return subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "warning",
             "-f", "rawvideo", "-pix_fmt", "rgba",
             "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", "-vf", "vflip",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH -- install ffmpeg to record video") from None
