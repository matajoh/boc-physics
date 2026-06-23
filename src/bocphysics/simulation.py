"""Module providing the simulation code."""

from colorsys import hls_to_rgb
import json
import random
import time

from bocpy import Matrix, pump, wait
import pyglet
from pyglet.window import key, mouse

from .bodies import Circle, Polygon
from .config import DetectionKind, Resolution
from .engine import PhysicsEngine, PhysicsMode
from .parallel import ParallelStepper
from .render import Camera, draw_frame
from .scene import DEFAULT_SCENE, Scene


class Simulation(pyglet.window.Window):
    """The interactive pyglet window driving the physics simulation."""

    def __init__(self, resolution: Resolution,
                 physics_mode=PhysicsMode.FRICTION,
                 detection_kind=DetectionKind.QUADTREE,
                 debug=False, show_contacts=False, snapshot=False,
                 scene: Scene = DEFAULT_SCENE,
                 parallel=False, workers=None):
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
        # counts physics steps actually taken, so a collapse can be pinned to a frame
        self.frame_count = 0
        self.fps_stats = "FPS: "
        self.physics_stats = "Physics: ms"

        self.engine = PhysicsEngine(resolution.width, resolution.height, physics_mode,
                                    detection_kind, show_contacts)
        self.camera = Camera(self.engine.center, self.engine.scale, resolution.height)
        for body in scene.build():
            self.engine.add_body(body)

        # parallel path: a pump-driven, depth-1 pipeline over BOC workers; the
        # serial engine.step stays the teaching baseline and the benchmark control
        self.parallel = parallel
        self.stepper = None
        self.in_flight = 0
        self.pending_spawns = []
        if self.parallel:
            self.stepper = ParallelStepper(self.engine)
            self.stepper.begin(worker_count=workers)

        self.labels = []
        if self.debug:
            # y-up means top-left labels sit near the window height
            for i in range(4):
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
        self.frame_shapes = draw_frame(self.engine.bodies, self.engine.contacts,
                                       self.batch, self.camera)
        self.batch.draw()
        if self.debug:
            self.labels[0].text = self.fps_stats
            self.labels[1].text = self.physics_stats
            self.labels[2].text = f"Bodies: {len(self.engine.bodies)}"
            self.labels[3].text = f"Frame: {self.frame_count}"
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
            self.spawn_body(Circle.create(radius, 2, color).move_to(pos))
        elif button == mouse.RIGHT:
            h = random.uniform(0, 1)
            r, g, b = hls_to_rgb(h, 0.5, 1.0)
            color = int(r * 255), int(g * 255), int(b * 255)
            if random.random() < 0.25:
                width = random.uniform(2, 3)
                height = random.uniform(2, 3)
                rectangle = Polygon.create_rectangle(width, height, 2, color)
                self.spawn_body(rectangle.move_to(pos))
            else:
                num_sides = random.randint(3, 8)
                radius = random.uniform(1.25, 1.5)
                polygon = Polygon.create_regular_polygon(num_sides, radius, 2, color)
                self.spawn_body(polygon.move_to(pos))

    def spawn_body(self, body):
        """Add a body now (serial) or queue it for the next frame boundary (parallel).

        Description:
            In the parallel path a frame may be in flight on the workers, which
            read the body set, so a spawn cannot mutate it mid-frame. The body is
            queued and applied once the pipeline is idle, which also bumps the
            geometry version so the next step re-seeds it.
        """
        if self.parallel:
            self.pending_spawns.append(body)
        else:
            self.engine.add_body(body)

    def on_close(self):
        """Save a snapshot if requested, drain any workers, then close the window."""
        if self.snapshot:
            self.save_snapshot()

        if self.parallel:
            wait()

        print("Shutting down...")
        self.close()

    def update(self, dt):
        """Advance the physics by one frame and refresh the debug stats."""
        if self.parallel:
            self.update_parallel(dt)
            return

        if not self.paused:
            start = time.perf_counter()
            self.engine.step(dt)
            self.physics_elapsed += time.perf_counter() - start
            self.frame_count += 1

        self.refresh_stats(dt)

    def refresh_stats(self, dt):
        """Accumulate one frame's timing and roll up the debug overlay strings."""
        self.frame_elapsed += dt
        self.samples += 1
        if self.samples > 100:
            self.fps_stats = f"FPS: {self.samples / self.frame_elapsed:.2f}"
            self.physics_stats = f"Physics: {self.physics_elapsed / self.samples * 1000:.2f} ms"
            self.physics_elapsed = 0
            self.frame_elapsed = 0
            self.samples = 0

    def update_parallel(self, dt):
        """Pump the previous parallel frame, then schedule the next when it is idle.

        Description:
            pump() runs any ready pinned behaviors and reports how many executed.
            Each frame schedules exactly one pinned writeback, so a non-zero count
            means the previous frame's writeback has landed and the pipeline is
            idle. Only then are queued spawns applied and the next frame scheduled,
            keeping at most one frame in flight (a depth-1 pipeline). The wall time
            spent driving the pipeline each tick is the parallel physics-cost proxy.
        """
        start = time.perf_counter()
        self.in_flight -= pump().executed
        if self.in_flight <= 0 and not self.paused:
            self.apply_pending_spawns()
            if self.stepper.step():
                self.in_flight += 1
                self.frame_count += 1
        self.physics_elapsed += time.perf_counter() - start

        self.refresh_stats(dt)

    def apply_pending_spawns(self):
        """Add any click-spawned bodies now that the parallel pipeline is idle."""
        for body in self.pending_spawns:
            self.engine.add_body(body)

        self.pending_spawns.clear()

    def run(self):
        """Schedule the update tick and enter the pyglet event loop."""
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        pyglet.app.run()
