"""Module providing declarative scene specifications.

A scene is a small, picklable description of the bodies that make up a
simulation -- the static parts (floors, walls, ledges), any dynamic bodies,
and the generators that emit a stream of dynamic bodies over time. The scene
is kept as data so that different arenas -- the default interactive scene, a
benchmark box, a pachinko board -- can run through the same engine and be
loaded from a file (see :meth:`Scene.from_dict`).
"""

from colorsys import hls_to_rgb
import json
import math
import random
from typing import List, NamedTuple, Tuple

from bocpy import Matrix

from .bodies import Circle, Polygon, RigidBody
from .render import Color


class BodySpec(NamedTuple):
    """A declarative spec for one body in a scene, static or dynamic."""

    kind: str
    color: Color
    position: Tuple[float, float]
    angle: float = 0.0
    width: float = 0.0
    height: float = 0.0
    num_sides: int = 0
    radius: float = 0.0
    density: float = 2.0

    def build(self, is_static: bool = True) -> RigidBody:
        """Construct the concrete rigid body from this spec."""
        if self.kind == "rectangle":
            body = Polygon.create_rectangle(self.width, self.height,
                                            self.density, self.color, is_static=is_static)
        elif self.kind == "regular_polygon":
            body = Polygon.create_regular_polygon(self.num_sides, self.radius,
                                                  self.density, self.color, is_static=is_static)
        elif self.kind == "circle":
            body = Circle.create(self.radius, self.density, self.color, is_static=is_static)
        else:
            raise ValueError(f"Unknown body kind: {self.kind}")

        body.move_to(Matrix.vector([self.position[0], self.position[1]]))
        body.rotate_to(self.angle)
        return body

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict describing this body."""
        return self._asdict()

    @staticmethod
    def from_dict(data: dict) -> "BodySpec":
        """Create a body spec from a dict, applying field defaults."""
        fields = {key: data[key] for key in BodySpec._fields if key in data}
        if "position" in fields:
            fields["position"] = tuple(fields["position"])
        if "color" in fields and isinstance(fields["color"], list):
            fields["color"] = tuple(fields["color"])
        return BodySpec(**fields)


class GeneratorSpec(NamedTuple):
    """A declarative emitter that drops dynamic bodies at a constant rate.

    Description:
        A generator is the dynamic counterpart to a BodySpec: where a BodySpec
        names one fixed body, a generator names a stream of bodies emitted from
        a point at a steady ``rate`` (bodies per second). Each emission jitters
        the spawn x by ``spread`` and the shape's size by ``size_jitter``, and
        optionally randomises the colour. ``limit`` caps the total emitted (0 is
        unlimited); ``seed`` makes the stream reproducible.
    """

    kind: str
    color: Color
    position: Tuple[float, float]
    rate: float
    radius: float = 0.0
    width: float = 0.0
    height: float = 0.0
    num_sides: int = 0
    density: float = 2.0
    spread: float = 0.0
    size_jitter: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    randomize_color: bool = False
    limit: int = 0
    seed: int = 0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict describing this generator."""
        return self._asdict()

    @staticmethod
    def from_dict(data: dict) -> "GeneratorSpec":
        """Create a generator spec from a dict, applying field defaults."""
        fields = {key: data[key] for key in GeneratorSpec._fields if key in data}
        if "position" in fields:
            fields["position"] = tuple(fields["position"])
        if "velocity" in fields:
            fields["velocity"] = tuple(fields["velocity"])
        if "color" in fields and isinstance(fields["color"], list):
            fields["color"] = tuple(fields["color"])
        return GeneratorSpec(**fields)


class Generator:
    """The runtime state that turns a GeneratorSpec into a stream of bodies.

    Description:
        The spec is immutable data; this object carries the per-run mutable
        state -- a seeded rng and a fractional accumulator -- so the same spec
        can drive several independent, reproducible streams. ``update(dt)``
        accrues ``rate * dt`` emissions and releases the whole-number part each
        frame, carrying the remainder so the long-run rate is exact.
    """

    def __init__(self, spec: GeneratorSpec):
        """Create a generator from its spec, seeding its rng and accumulator."""
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self.accumulator = 0.0
        self.emitted = 0

    def emit(self) -> RigidBody:
        """Build one jittered dynamic body from the spec via a transient BodySpec."""
        spec = self.spec
        x = spec.position[0] + self.rng.uniform(-spec.spread, spec.spread)
        factor = 1.0 + self.rng.uniform(-spec.size_jitter, spec.size_jitter)
        color = spec.color
        if spec.randomize_color:
            r, g, b = hls_to_rgb(self.rng.random(), 0.5, 1.0)
            color = (int(r * 255), int(g * 255), int(b * 255))
        body = BodySpec(spec.kind, color, (x, spec.position[1]),
                        width=spec.width * factor, height=spec.height * factor,
                        num_sides=spec.num_sides, radius=spec.radius * factor,
                        density=spec.density).build(is_static=False)
        if spec.velocity != (0.0, 0.0):
            body.linear_velocity = Matrix.vector([spec.velocity[0], spec.velocity[1]])
        return body

    def update(self, dt: float) -> List[RigidBody]:
        """Advance the accumulator by dt and return the bodies emitted this frame."""
        self.accumulator += self.spec.rate * dt
        bodies = []
        while self.accumulator >= 1.0:
            if self.spec.limit and self.emitted >= self.spec.limit:
                self.accumulator = 0.0
                break
            self.accumulator -= 1.0
            bodies.append(self.emit())
            self.emitted += 1
        return bodies


class Scene(NamedTuple):
    """A named collection of static and dynamic bodies and emitters."""

    name: str
    statics: Tuple[BodySpec, ...]
    dynamics: Tuple[BodySpec, ...] = ()
    generators: Tuple[GeneratorSpec, ...] = ()
    view_height: float = 0.0
    view_aspect: float = 0.0

    def build(self) -> List[RigidBody]:
        """Construct all of the scene's bodies, statics first then dynamics."""
        return ([spec.build(is_static=True) for spec in self.statics] +
                [spec.build(is_static=False) for spec in self.dynamics])

    def make_generators(self) -> List[Generator]:
        """Instantiate a fresh runtime Generator for each of the scene's emitters."""
        return [Generator(spec) for spec in self.generators]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict describing this scene."""
        return {"name": self.name,
                "statics": [spec.to_dict() for spec in self.statics],
                "dynamics": [spec.to_dict() for spec in self.dynamics],
                "generators": [spec.to_dict() for spec in self.generators],
                "view_height": self.view_height,
                "view_aspect": self.view_aspect}

    @staticmethod
    def from_dict(data: dict) -> "Scene":
        """Create a scene from a dict produced by :meth:`to_dict`."""
        statics = tuple(BodySpec.from_dict(spec) for spec in data["statics"])
        dynamics = tuple(BodySpec.from_dict(spec) for spec in data.get("dynamics", ()))
        generators = tuple(GeneratorSpec.from_dict(spec)
                           for spec in data.get("generators", ()))
        return Scene(data["name"], statics, dynamics, generators,
                     data.get("view_height", 0.0), data.get("view_aspect", 0.0))

    def save(self, path: str):
        """Write this scene to a JSON file."""
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4)

    @staticmethod
    def load(name: str) -> "Scene":
        """Load a scene by built-in name, or from a JSON file path."""
        if name in BUILTIN_SCENES:
            return BUILTIN_SCENES[name]

        with open(name, "r", encoding="utf-8") as file:
            return Scene.from_dict(json.load(file))


DEFAULT_SCENE = Scene("default", (
    BodySpec("rectangle", "darkgreen", (0, 10), width=30, height=2),
    BodySpec("rectangle", "gray", (-7.5, 0), angle=math.pi / 10, width=15, height=1),
    BodySpec("rectangle", "darkred", (7, -5), angle=-math.pi / 10, width=10, height=1),
))

OPEN_BOX = Scene("open_box", (
    BodySpec("rectangle", "darkgreen", (0, 10), width=30, height=2),
    BodySpec("rectangle", "gray", (-14, -2), width=2, height=24),
    BodySpec("rectangle", "gray", (14, -2), width=2, height=24),
))

STACK_BOX_SIZE = 2.0
STACK_FLOOR_TOP = 9.0
GOLDEN_SEED = 20260608


def floor_spec() -> BodySpec:
    """The wide green static floor shared by the benchmark scenes."""
    return BodySpec("rectangle", (0, 100, 0), (0, 10), width=30, height=2)


def make_stack_scene(levels: int = 8) -> Scene:
    """Build a floor with a vertical column of equal boxes resting on it."""
    dynamics = []
    for i in range(levels):
        y = STACK_FLOOR_TOP - STACK_BOX_SIZE / 2 - i * STACK_BOX_SIZE
        dynamics.append(BodySpec("rectangle", (50, 120, 200), (0, y),
                                 width=STACK_BOX_SIZE, height=STACK_BOX_SIZE))
    return Scene("stack", (floor_spec(),), tuple(dynamics))


def make_pyramid_scene(levels: int = 8) -> Scene:
    """Build a brick pyramid where each upper box straddles two below."""
    dynamics = []
    for row in range(levels):
        y = STACK_FLOOR_TOP - STACK_BOX_SIZE / 2 - row * STACK_BOX_SIZE
        count = levels - row
        offset = (count - 1) * STACK_BOX_SIZE / 2
        for col in range(count):
            x = col * STACK_BOX_SIZE - offset
            dynamics.append(BodySpec("rectangle", (50, 120, 200), (x, y),
                                     width=STACK_BOX_SIZE, height=STACK_BOX_SIZE))
    return Scene("pyramid", (floor_spec(),), tuple(dynamics))


def make_golden_scene(seed: int = GOLDEN_SEED) -> Scene:
    """Build the deterministic seeded scatter the golden-master oracle settles.

    Description:
        The rng draw order (x, y, angle, kind, then the shape's own dimensions)
        is byte-for-byte the order the engine's golden-master builder used, so a
        scene built here is bit-identical to the recorded oracle scene.
    """
    rng = random.Random(seed)
    dynamics = []
    for _ in range(24):
        x = rng.uniform(-12, 12)
        y = rng.uniform(-12, 6)
        angle = rng.uniform(0, 6.28)
        kind = rng.random()
        if kind < 0.4:
            spec = BodySpec("circle", (200, 100, 50), (x, y), angle=angle,
                            radius=rng.uniform(0.6, 1.2))
        elif kind < 0.7:
            spec = BodySpec("rectangle", (50, 120, 200), (x, y), angle=angle,
                            width=rng.uniform(1.2, 2.2), height=rng.uniform(1.2, 2.2))
        else:
            spec = BodySpec("regular_polygon", (180, 60, 160), (x, y), angle=angle,
                            num_sides=rng.randint(3, 6), radius=rng.uniform(0.8, 1.3))
        dynamics.append(spec)
    statics = (floor_spec(),
               BodySpec("rectangle", (120, 120, 120), (-14, -2), width=2, height=24),
               BodySpec("rectangle", (120, 120, 120), (14, -2), width=2, height=24))
    return Scene("golden", statics, tuple(dynamics))


def _bar_between(start, end, thickness: float, color: Color) -> BodySpec:
    """Return a rectangle bar spanning two points, padded so its ends overlap."""
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    return BodySpec("rectangle", color, mid, angle=angle,
                    width=length + thickness, height=thickness)


def make_pachinko_scene(rows: int = 6) -> Scene:
    """Build a classic pachinko board: a peg grid feeding five lanes that drain.

    Parametric in ``rows``: the vertical extent and zoom scale to fit; the
    horizontal layout (walls, lanes, funnel) is a fixed five-lane board.
    """
    x_wall = 10.0
    peg_step, spacing = 2.5, 3.0
    peg_span = (rows - 1) * peg_step
    machine_height = 18.0 + peg_span
    wall_top = -machine_height / 2
    wall_bottom = wall_top + machine_height
    wall_mid = (wall_top + wall_bottom) / 2
    statics = [BodySpec("rectangle", "gray", (-x_wall, wall_mid), width=1.0, height=machine_height),
               BodySpec("rectangle", "gray", (x_wall, wall_mid), width=1.0, height=machine_height)]
    peg_top = wall_top + 3.0
    for row in range(rows):
        y = peg_top + row * peg_step
        if row % 2 == 0:
            xs = [k * spacing for k in range(-2, 3)]
        else:
            xs = [(k + 0.5) * spacing for k in range(-3, 3)]
        for x in xs:
            statics.append(BodySpec("circle", "slategray", (x, y), radius=0.35))
    lane_top = peg_top + peg_span + 3.5
    lane_bottom = lane_top + 4.0
    lane_mid, lane_span = (lane_top + lane_bottom) / 2, lane_bottom - lane_top
    for x in (-6.0, -3.0, 0.0, 3.0, 6.0):
        statics.append(BodySpec("rectangle", "gray", (x, lane_mid), width=0.6, height=lane_span))
    floor_outer, floor_inner = lane_bottom + 1.0, lane_bottom + 4.0
    statics.append(_bar_between((-x_wall, floor_outer), (-3.0, floor_inner), 0.6, "gray"))
    statics.append(_bar_between((x_wall, floor_outer), (3.0, floor_inner), 0.6, "gray"))
    statics.append(BodySpec("rectangle", "gray", (0.0, wall_bottom), width=x_wall - 0.5, height=0.6))
    generator = GeneratorSpec("circle", (0, 0, 0), (0.0, wall_top + 0.5), rate=3.0,
                              radius=0.6, spread=4.0, randomize_color=True)
    view_height = machine_height + 5.5
    return Scene("pachinko", tuple(statics), (), (generator,),
                 view_height=view_height, view_aspect=24.0 / view_height)


BUILTIN_SCENES = {scene.name: scene for scene in (
    DEFAULT_SCENE, OPEN_BOX, make_stack_scene(), make_pyramid_scene(),
    make_golden_scene(), make_pachinko_scene())}
