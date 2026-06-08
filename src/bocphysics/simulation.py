"""Module providing the simulation code."""

from colorsys import hls_to_rgb
import json
import random
import time

from bocpy import Matrix
import pyglet
from pyglet.window import key, mouse

from .bodies import Circle, Polygon
from .config import DetectionKind, Resolution
from .engine import PhysicsEngine, PhysicsMode
from .render import Camera
from .scene import DEFAULT_SCENE, Scene


class Simulation(pyglet.window.Window):
    """The interactive pyglet window driving the physics simulation."""

    def __init__(self, resolution: Resolution,
                 physics_mode=PhysicsMode.FRICTION,
                 detection_kind=DetectionKind.QUADTREE,
                 debug=False, show_contacts=False, snapshot=False,
                 scene: Scene = DEFAULT_SCENE):
        """Create the window, physics engine, and the static scene bodies."""
        super().__init__(resolution.width, resolution.height, "bocphysics")
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.debug = debug
        self.snapshot = snapshot
        self.paused = False
        self.batch = pyglet.graphics.Batch()
        # shapes are kept alive on self so they survive until batch.draw()
        self.frame_shapes = []
        self.physics_elapsed = 0
        self.frame_elapsed = 0
        self.samples = 0
        self.fps_stats = "FPS: "
        self.physics_stats = "Physics: ms"

        self.engine = PhysicsEngine(resolution.width, resolution.height, physics_mode,
                                    detection_kind, show_contacts)
        self.camera = Camera(self.engine.center, self.engine.scale, resolution.height)
        for body in scene.build():
            self.engine.add_body(body)

        self.labels = []
        if self.debug:
            # y-up means top-left labels sit near the window height
            for i in range(3):
                self.labels.append(pyglet.text.Label("", font_size=18, x=10,
                                                     y=resolution.height - 20 - i * 20,
                                                     color=(0, 0, 0, 255)))

    def save_snapshot(self):
        """Save a PNG screenshot and a JSON dump of the current bodies."""
        path = f"snapshot_{time.strftime('%Y%m%d-%H%M%S')}"
        pyglet.image.get_buffer_manager().get_color_buffer().save(f"{path}.png")
        bodies = [body.to_dict() for body in self.engine.bodies]
        with open(f"{path}.json", "w", encoding="utf-8") as file:
            json.dump(bodies, file, indent=4)

    def on_draw(self):
        """Rebuild the frame's shapes into a fresh batch and draw them."""
        self.clear()
        self.batch = pyglet.graphics.Batch()
        self.frame_shapes = self.engine.draw(self.batch, self.camera)
        self.batch.draw()
        if self.debug:
            self.labels[0].text = self.fps_stats
            self.labels[1].text = self.physics_stats
            self.labels[2].text = f"Bodies: {len(self.engine.bodies)}"
            for label in self.labels:
                label.draw()

    def on_key_press(self, symbol, modifiers):
        """Close on ESCAPE and toggle the pause state on SPACE."""
        if symbol == key.ESCAPE:
            self.on_close()
        elif symbol == key.SPACE:
            self.paused = not self.paused

    def on_mouse_press(self, x, y, button, modifiers):
        """Spawn a circle on left-click and a polygon on right-click."""
        # convert pyglet (y-up) pixels back to the engine's y-down world
        pos = self.engine.to_world(Matrix.vector([x, self.height - y]))
        if button == mouse.LEFT:
            radius = random.uniform(1, 1.25)
            h = random.uniform(0, 1)
            r, g, b = hls_to_rgb(h, 0.5, 1.0)
            color = int(r * 255), int(g * 255), int(b * 255)
            self.engine.add_body(Circle.create(radius, 2, color).move_to(pos))
        elif button == mouse.RIGHT:
            h = random.uniform(0, 1)
            r, g, b = hls_to_rgb(h, 0.5, 1.0)
            color = int(r * 255), int(g * 255), int(b * 255)
            if random.random() < 0.25:
                width = random.uniform(2, 3)
                height = random.uniform(2, 3)
                rectangle = Polygon.create_rectangle(width, height, 2, color)
                self.engine.add_body(rectangle.move_to(pos))
            else:
                num_sides = random.randint(3, 8)
                radius = random.uniform(1.25, 1.5)
                polygon = Polygon.create_regular_polygon(num_sides, radius, 2, color)
                self.engine.add_body(polygon.move_to(pos))

    def on_close(self):
        """Save a snapshot if requested, then close the window."""
        if self.snapshot:
            self.save_snapshot()

        print("Shutting down...")
        self.close()

    def update(self, dt):
        """Advance the physics by one frame and refresh the debug stats."""
        if not self.paused:
            start = time.perf_counter()
            self.engine.step(dt)
            self.physics_elapsed += time.perf_counter() - start

        self.frame_elapsed += dt
        self.samples += 1
        if self.samples > 100:
            self.fps_stats = f"FPS: {self.samples / self.frame_elapsed:.2f}"
            self.physics_stats = f"Physics: {self.physics_elapsed / self.samples * 1000:.2f} ms"
            self.physics_elapsed = 0
            self.frame_elapsed = 0
            self.samples = 0

    def run(self):
        """Schedule the update tick and enter the pyglet event loop."""
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        pyglet.app.run()
