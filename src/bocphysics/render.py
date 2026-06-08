"""Module providing rendering helpers shared by the bodies and engine.

This module isolates the pyglet rendering details (colour conversion and
the world-to-screen projection) behind a small seam so the physics code
never imports pyglet directly.
"""

from typing import Tuple, Union

from bocpy import Matrix
import webcolors


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
