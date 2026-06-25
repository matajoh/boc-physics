"""Tests for the rendering helpers (colour conversion and projection)."""

import math
import random

from bocpy import Matrix
import pytest

from bocphysics.render import BLACK, Camera, open_encoder, to_grayscale, to_rgba, YELLOW


def _pyglet_graphics_available() -> bool:
    """True when pyglet's graphics backend imports and a Group can be built (needs a display/GL)."""
    try:
        from pyglet import graphics
        graphics.Group(order=0)
    except Exception:
        return False
    return True


requires_pyglet_graphics = pytest.mark.skipif(
    not _pyglet_graphics_available(),
    reason="pyglet graphics backend unavailable (headless CI: no display/GL context)",
)


@requires_pyglet_graphics
def test_draw_static_layer_orders_each_body_fill_below_its_outline(monkeypatch):
    """Each static gets an increasing fill/outline order so fills paint over prior outlines."""
    from bocphysics import render

    captured = []

    def fake_draw_body(body, batch, project, fill_group, line_group):
        captured.append((fill_group.order, line_group.order))
        return ()

    monkeypatch.setattr(render, "draw_body", fake_draw_body)
    render.draw_static_layer([object(), object(), object()], batch=None, project=None)

    assert captured == [(0, 1), (2, 3), (4, 5)]


def test_to_rgba_named_color():
    assert to_rgba("darkgreen") == (0, 100, 0, 255)
    assert to_rgba("black") == (0, 0, 0, 255)
    assert to_rgba("white") == (255, 255, 255, 255)


def test_to_rgba_rgb_tuple_gets_alpha():
    assert to_rgba((10, 20, 30)) == (10, 20, 30, 255)


def test_to_rgba_rgba_tuple_passthrough():
    assert to_rgba((10, 20, 30, 128)) == (10, 20, 30, 128)


def test_to_grayscale_collapses_to_luminance():
    assert to_grayscale("black") == (0, 0, 0, 255)
    assert to_grayscale("white") == (255, 255, 255, 255)
    assert to_grayscale((200, 200, 200)) == (200, 200, 200, 255)


def test_to_grayscale_preserves_alpha():
    r, g, b, a = to_grayscale((10, 20, 30, 128))
    assert (r, g, b) == (18, 18, 18)
    assert a == 128


def test_color_constants():
    assert BLACK == (0, 0, 0, 255)
    assert YELLOW == (255, 255, 0, 255)


def test_camera_center_maps_to_screen_center():
    cam = Camera(Matrix.vector([20, 15]), 30.0, 900.0)
    assert cam(Matrix.vector([0, 0])) == (600.0, 450.0)


def test_camera_flips_y_axis():
    cam = Camera(Matrix.vector([20, 15]), 30.0, 900.0)
    top = cam(Matrix.vector([0, -15]))
    bottom = cam(Matrix.vector([0, 15]))
    assert bottom[1] < top[1]


def test_camera_projection_formula_fuzz():
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


def test_open_encoder_builds_ffmpeg_command(monkeypatch):
    import subprocess

    captured = {}

    class FakePopen:
        def __init__(self, args, stdin=None):
            captured["args"] = args
            captured["stdin"] = stdin

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    open_encoder("out.mp4", 640, 480, 30)
    args = captured["args"]
    assert args[0] == "ffmpeg"
    assert "640x480" in args
    assert "30" in args
    assert "out.mp4" in args
    assert captured["stdin"] is subprocess.PIPE


def test_open_encoder_missing_ffmpeg_raises_runtime_error(monkeypatch):
    """A missing ffmpeg yields a catchable RuntimeError, not a process exit."""
    import subprocess

    def fake_popen(*args, **kwargs):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        open_encoder("out.mp4", 640, 480, 30)
