"""Module providing the simulation code."""

from colorsys import hls_to_rgb
import json
import os
import random
import time

from bocpy import Matrix, pump, quiesce, wait
import pyglet
from pyglet.window import key, mouse

from .bodies import Circle, Polygon
from .config import DetectionKind, Resolution
from .engine import PhysicsEngine
from .parallel import AUTO_SLABS, MIN_SLAB_BODIES, ParallelStepper, resolve_slab_count
from .patches import build_slab_partition, slab_boundaries
from .quadtree import QuadTree
from .render import (Camera, draw_box_overlay, draw_frame, draw_slab_fills,
                     draw_static_layer)
from .scene import DEFAULT_SCENE, Scene
from .spawn import SpawnQueue

# Cap the physics step so a slow render or startup frame cannot blow up the explicit integrator.
MAX_PHYSICS_DT = 1 / 50


def _fit_resolution(resolution: Resolution, view_aspect: float) -> Resolution:
    """Reshape the window to a scene's world aspect, keeping even, MP4-safe sides."""
    if not view_aspect:
        return resolution
    width = max(2, int(round(resolution.height * view_aspect / 2)) * 2)
    return Resolution(width, resolution.height)


class Simulation(pyglet.window.Window):
    """The interactive pyglet window driving the physics simulation."""

    def __init__(self, resolution: Resolution,
                 detection_kind=DetectionKind.QUADTREE,
                 debug=False, show_contacts=False, snapshot=False,
                 scene: Scene = DEFAULT_SCENE,
                 parallel=False, workers=None, overlay="none", visible=True):
        """Create the window, physics engine, and the static scene bodies."""
        resolution = _fit_resolution(resolution, scene.view_aspect)
        super().__init__(resolution.width, resolution.height, "bocphysics",
                         visible=visible)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.debug = debug
        self.snapshot = snapshot
        self.paused = False
        self.overlay = overlay
        self.batch = pyglet.graphics.Batch()
        self.frame_shapes = []
        self.physics_elapsed = 0
        self.frame_elapsed = 0
        self.samples = 0
        self.frame_count = 0
        self.last_frame_count = 0
        self.fps_stats = "FPS: "
        self.physics_stats = "Physics: ms"

        view_height = scene.view_height or 30
        self.engine = PhysicsEngine(resolution.width, resolution.height,
                                    detection_kind, show_contacts,
                                    height_in_meters=view_height)
        self.camera = Camera(self.engine.center, self.engine.scale, resolution.height)
        for body in scene.build():
            self.engine.add_body(body)

        self.generators = scene.make_generators()

        self.static_batch = pyglet.graphics.Batch()
        statics = [body for body in self.engine.bodies if body.render and not body.physics]
        self.static_shapes = draw_static_layer(statics, self.static_batch, self.camera)

        self.parallel = parallel
        self.stepper = None
        self.in_flight = 0
        self.spawn_queue = SpawnQueue()
        if self.parallel:
            self.stepper = ParallelStepper(self.engine)
            self.stepper.begin(worker_count=workers)

        self.labels = []
        if self.debug:
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
        self.static_batch.draw()
        self.batch = pyglet.graphics.Batch()
        self.frame_shapes = self.render_scene(self.batch)
        self.batch.draw()
        if self.debug:
            self.labels[0].text = self.fps_stats
            self.labels[1].text = self.physics_stats
            self.labels[2].text = f"Bodies: {len(self.engine.bodies)}"
            self.labels[3].text = f"Frame: {self.frame_count}"
            for label in self.labels:
                label.draw()

    def render_scene(self, batch):
        """Build the moving bodies, contacts, and any partition overlay into one batch."""
        grayscale = self.overlay != "none"
        dynamics = [body for body in self.engine.bodies if body.render and body.physics]
        kept = draw_frame(dynamics, self.engine.contacts, batch, self.camera, grayscale)
        kept.extend(self.draw_overlay(batch))
        return kept

    def render_to_buffer(self):
        """Draw one offscreen frame (statics then dynamics) and return the kept shapes and colour buffer."""
        self.switch_to()
        self.clear()
        self.static_batch.draw()
        batch = pyglet.graphics.Batch()
        kept = self.render_scene(batch)
        batch.draw()
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        return kept, buffer

    def draw_overlay(self, batch):
        """Build the selected partition overlay (slabs or quadtree) into a batch."""
        box = self.engine.detection.box
        if self.overlay == "slabs":
            count = self.stepper.num_slabs if self.stepper else resolve_slab_count(AUTO_SLABS, None)
            partition = build_slab_partition(self.engine.bodies, [], box,
                                             count, min_slab_bodies=MIN_SLAB_BODIES)
            edges = [box.left, *slab_boundaries(partition), box.right]
            return draw_slab_fills(edges, box.top, box.bottom, batch, self.camera)
        if self.overlay == "quadtree":
            tree = QuadTree(box)
            for body in self.engine.bodies:
                if body.collision and hasattr(body, "swept_aabb"):
                    tree.add(body)
            return draw_box_overlay(tree.boxes(), batch, self.camera)

        return []

    def on_key_press(self, symbol, modifiers):
        """Close on ESCAPE and toggle the pause state on SPACE."""
        if symbol == key.ESCAPE:
            self.on_close()
        elif symbol == key.SPACE:
            self.paused = not self.paused

    def on_mouse_press(self, x, y, button, modifiers):
        """Spawn a circle on left-click and a polygon on right-click."""
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
        """Queue a runtime-spawned body for admission once it fits without overlap.

        Description:
            A click or generator can drop a body straight onto the pile; entering
            the world at a deep overlap makes the solver fling it out. The queue
            holds it until a frame where it fits, or discards it after a budget of
            tries, keeping spawn admission entirely outside the physics step.
        """
        self.spawn_queue.enqueue(body)

    def admit_spawns(self):
        """Add every queued spawn that now fits without significant overlap."""
        for body in self.spawn_queue.process(self.engine.bodies):
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
            step_dt = min(dt, MAX_PHYSICS_DT)
            self.tick_generators(step_dt)
            self.admit_spawns()
            start = time.perf_counter()
            self.engine.step(step_dt)
            self.physics_elapsed += time.perf_counter() - start
            self.frame_count += 1

        self.refresh_stats(dt)

    def tick_generators(self, dt):
        """Emit this frame's generated bodies, routing each through spawn_body."""
        for generator in self.generators:
            for body in generator.update(dt):
                self.spawn_body(body)

    def refresh_stats(self, dt):
        """Accumulate one frame's timing and roll up the debug overlay strings."""
        self.frame_elapsed += dt
        self.samples += 1
        if self.samples > 100:
            self.fps_stats = f"FPS: {self.samples / self.frame_elapsed:.2f}"
            if self.parallel:
                physics_frames = self.frame_count - self.last_frame_count
                ms = self.frame_elapsed / physics_frames * 1000 if physics_frames else 0.0
            else:
                ms = self.physics_elapsed / self.samples * 1000
            self.physics_stats = f"Physics: {ms:.2f} ms"
            self.last_frame_count = self.frame_count
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
            keeping at most one frame in flight (a depth-1 pipeline). The worker
            solve runs async, so the honest per-frame cost is the rate at which
            frames actually complete, not the main-thread time spent here.
        """
        self.in_flight -= pump().executed
        if self.in_flight <= 0 and not self.paused:
            self.tick_generators(dt)
            self.admit_spawns()
            if self.stepper.step():
                self.in_flight += 1
                self.frame_count += 1

        self.refresh_stats(dt)

    def run(self):
        """Schedule the update tick and enter the pyglet event loop."""
        pyglet.clock.schedule_interval(self.update, 1 / 60)
        pyglet.app.run()

    def step_once(self, dt):
        """Advance the physics one frame, draining the parallel pipeline if running."""
        self.tick_generators(dt)
        self.admit_spawns()
        if self.parallel:
            if self.stepper.step():
                quiesce(30.0)
        else:
            self.engine.step(dt)
        self.frame_count += 1

    def record(self, path: str, frames: int, fps: int, dt: float = 1 / 60):
        """Render a fixed number of frames straight to an mp4 file via ffmpeg.

        Description:
            Unlike the interactive loop, this advances at a fixed dt and captures
            every frame, so the output is deterministic and independent of the
            wall clock. Generators, the serial or parallel step, and the overlay
            all run exactly as they do on screen. The generator-fed scenes let
            the drop-box benchmark be recorded straight from the simulator.
        """
        from .render import open_encoder

        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.switch_to()
        self.clear()
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        encoder = open_encoder(path, buffer.width, buffer.height, fps)
        try:
            for _ in range(frames):
                self.step_once(dt)
                kept, buffer = self.render_to_buffer()
                data = buffer.get_image_data().get_data("RGBA", buffer.width * 4)
                encoder.stdin.write(data)
                del kept
        finally:
            encoder.stdin.close()
            encoder.wait()
            if self.parallel:
                wait()
            self.close()

        print(f"wrote {path} ({frames} frames at {fps} fps, "
              f"{len(self.engine.bodies)} bodies)")
