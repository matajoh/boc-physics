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


def draw_body(body, batch, project: Camera) -> Tuple:
    """Draw one body into the batch by type, returning the shapes to keep alive."""
    from pyglet import shapes
    if isinstance(body, Circle):
        x, y = project(body.position)
        radius = body.radius * project.scale
        p = Matrix.vector([math.cos(body.angle), math.sin(body.angle)]) * radius * 0.9
        # the screen is y-down once projected, so the heading flips its y
        fill = shapes.Circle(x, y, radius, color=to_rgba(body.color), batch=batch)
        outline = shapes.Arc(x, y, radius, thickness=4, color=BLACK, batch=batch)
        heading = shapes.Line(x, y, x + p.x, y - p.y, thickness=4, color=BLACK, batch=batch)
        return (fill, outline, heading)

    # the only other body type is a polygon, drawn from its world-space vertices
    vertices = [project(v) for v in body.transformed_vertices]
    fill = shapes.Polygon(*vertices, color=to_rgba(body.color), batch=batch)
    outline = shapes.MultiLine(*vertices, closed=True, thickness=4, color=BLACK, batch=batch)
    return (fill, outline)


def draw_frame(bodies, contacts, batch, project: Camera) -> list:
    """Draw every renderable body and contact point; return the shapes to keep alive."""
    from pyglet import shapes
    kept = []
    for body in bodies:
        if body.render:
            kept.extend(draw_body(body, batch, project))

    for contact in contacts:
        x, y = project(Matrix.vector([contact[0], contact[1]]))
        kept.append(shapes.Circle(x, y, 5, color=YELLOW, batch=batch))
        kept.append(shapes.Arc(x, y, 5, thickness=2, color=BLACK, batch=batch))

    return kept
