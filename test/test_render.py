"""Tests for the rendering helpers (colour conversion and projection)."""

import math
import random

from bocpy import Matrix

from bocphysics.render import BLACK, Camera, to_rgba, YELLOW


def test_to_rgba_named_color():
    assert to_rgba("darkgreen") == (0, 100, 0, 255)
    assert to_rgba("black") == (0, 0, 0, 255)
    assert to_rgba("white") == (255, 255, 255, 255)


def test_to_rgba_rgb_tuple_gets_alpha():
    assert to_rgba((10, 20, 30)) == (10, 20, 30, 255)


def test_to_rgba_rgba_tuple_passthrough():
    assert to_rgba((10, 20, 30, 128)) == (10, 20, 30, 128)


def test_color_constants():
    assert BLACK == (0, 0, 0, 255)
    assert YELLOW == (255, 255, 0, 255)


def test_camera_center_maps_to_screen_center():
    # center=(20, 15), scale=30, height=900: world origin -> (600, 450)
    cam = Camera(Matrix.vector([20, 15]), 30.0, 900.0)
    assert cam(Matrix.vector([0, 0])) == (600.0, 450.0)


def test_camera_flips_y_axis():
    # world +y is "down"; after the y-up flip it must move toward y=0
    cam = Camera(Matrix.vector([20, 15]), 30.0, 900.0)
    top = cam(Matrix.vector([0, -15]))
    bottom = cam(Matrix.vector([0, 15]))
    assert bottom[1] < top[1]


def test_camera_projection_formula_fuzz():
    # the projection must match (x+cx)*s, h-(y+cy)*s for any inputs
    rng = random.Random(1234)
    for _ in range(200):
        cx = rng.uniform(-50, 50)
        cy = rng.uniform(-50, 50)
        scale = rng.uniform(1, 100)
        height = rng.uniform(100, 2000)
        px = rng.uniform(-100, 100)
        py = rng.uniform(-100, 100)
        cam = Camera(Matrix.vector([cx, cy]), scale, height)
        sx, sy = cam(Matrix.vector([px, py]))
        assert math.isclose(sx, (px + cx) * scale)
        assert math.isclose(sy, height - (py + cy) * scale)
