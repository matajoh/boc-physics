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


class ShapeCategory(NamedTuple):
    """One outcome of a generator's categorical shape distribution.

    Description:
        ``kind`` is one of ``circle``, ``rectangle`` or ``regular_polygon``;
        ``weight`` is its relative probability (weights are normalised across a
        generator's categories); ``num_sides`` is used only by polygons.
    """

    kind: str
    weight: float = 1.0
    num_sides: int = 0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict describing this category."""
        return self._asdict()

    @staticmethod
    def from_dict(data: dict) -> "ShapeCategory":
        """Create a shape category from a dict, applying field defaults."""
        fields = {key: data[key] for key in ShapeCategory._fields if key in data}
        return ShapeCategory(**fields)


class GeneratorSpec(NamedTuple):
    """A declarative emitter that drops dynamic bodies by sampling distributions.

    Description:
        Each emission samples four independent quantities: the spawn ``x`` from
        ``Uniform(x_range)``, the spawn ``y`` from ``Uniform(y_range)``, a shape
        from the categorical ``shapes`` distribution, and a ``size`` (a diameter)
        from ``Uniform(size_range)``. A circle or polygon takes radius
        ``size / 2``; a rectangle draws ``size`` twice for its width and height.
        Emissions follow a Poisson process at mean ``rate`` per second, so the
        gaps between drops are exponentially distributed (memoryless timing
        jitter). ``randomize_color`` gives each body a random hue, otherwise the
        fixed ``color`` is used; ``limit`` caps the total emitted (0 is
        unlimited); ``seed`` makes the stream reproducible.
    """

    shapes: Tuple[ShapeCategory, ...]
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    size_range: Tuple[float, float]
    rate: float
    color: Color = (200, 200, 200)
    randomize_color: bool = False
    density: float = 2.0
    limit: int = 0
    seed: int = 0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict describing this generator."""
        data = self._asdict()
        data["shapes"] = [shape.to_dict() for shape in self.shapes]
        return data

    @staticmethod
    def from_dict(data: dict) -> "GeneratorSpec":
        """Create a generator spec from a dict, applying field defaults."""
        fields = {key: data[key] for key in GeneratorSpec._fields if key in data}
        if "shapes" in fields:
            fields["shapes"] = tuple(ShapeCategory.from_dict(s) for s in fields["shapes"])
        for key in ("x_range", "y_range", "size_range"):
            if key in fields:
                fields[key] = tuple(fields[key])
        if "color" in fields and isinstance(fields["color"], list):
            fields["color"] = tuple(fields["color"])
        return GeneratorSpec(**fields)


class Generator:
    """The runtime state that turns a GeneratorSpec into a stream of bodies.

    Description:
        The spec is immutable data; this object carries the per-run mutable
        state -- a seeded rng and the time of the next scheduled emission -- so
        the same spec can drive several independent, reproducible streams.
        ``update(dt)`` advances a clock and releases every body whose Poisson
        arrival time has passed this frame.
    """

    def __init__(self, spec: GeneratorSpec):
        """Create a generator from its spec, seeding its rng and arrival clock."""
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self.emitted = 0
        self.clock = 0.0
        self.next_time = self._next_interval()

    def _next_interval(self) -> float:
        """Draw an exponential gap until the next emission, or never if idle."""
        if self.spec.rate <= 0:
            return math.inf
        return self.rng.expovariate(self.spec.rate)

    def _sample_size(self) -> float:
        """Sample one diameter from the uniform size distribution."""
        return self.rng.uniform(*self.spec.size_range)

    def _sample_color(self) -> Color:
        """Return a random hue when enabled, otherwise the spec's fixed colour."""
        if self.spec.randomize_color:
            r, g, b = hls_to_rgb(self.rng.random(), 0.5, 1.0)
            return (int(r * 255), int(g * 255), int(b * 255))
        return self.spec.color

    def emit(self) -> RigidBody:
        """Sample a position, shape, size and colour, and build one dynamic body."""
        spec = self.spec
        shape = self.rng.choices(spec.shapes, weights=[s.weight for s in spec.shapes])[0]
        x = self.rng.uniform(*spec.x_range)
        y = self.rng.uniform(*spec.y_range)
        color = self._sample_color()
        if shape.kind == "rectangle":
            body = BodySpec("rectangle", color, (x, y), width=self._sample_size(),
                            height=self._sample_size(), density=spec.density)
        elif shape.kind == "circle":
            body = BodySpec("circle", color, (x, y), radius=self._sample_size() / 2,
                            density=spec.density)
        else:
            body = BodySpec("regular_polygon", color, (x, y), num_sides=shape.num_sides,
                            radius=self._sample_size() / 2, density=spec.density)
        return body.build(is_static=False)

    def update(self, dt: float) -> List[RigidBody]:
        """Advance the arrival clock by dt and return the bodies emitted this frame."""
        self.clock += dt
        bodies = []
        while self.clock >= self.next_time:
            if self.spec.limit and self.emitted >= self.spec.limit:
                self.next_time = math.inf
                break
            bodies.append(self.emit())
            self.emitted += 1
            self.next_time += self._next_interval()
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


def sample_free_position(rng, placed, bound, attempts=4000):
    """Rejection-sample a spawn centre clear of every already-placed shape.

    Description:
        Draws (x, y) inside the scatter column until the disc of the given
        bounding radius clears every placed shape by a small margin, so no two
        bodies start interpenetrating. Falls back to the last draw if the
        attempt budget is exhausted (the domain is far from that crowded).
    """
    margin = 0.3
    x = y = 0.0
    for _ in range(attempts):
        x = rng.uniform(-11.3, 11.3)
        y = rng.uniform(-12, 6)
        if all((x - px) ** 2 + (y - py) ** 2 >= (bound + pb + margin) ** 2
               for px, py, pb in placed):
            break
    return x, y


def make_golden_scene(seed: int = GOLDEN_SEED) -> Scene:
    """Build the deterministic seeded scatter the golden-master oracle settles.

    Description:
        Shapes are rejection-sampled so none overlaps another at spawn -- deep
        initial penetration would otherwise fling overlapping bodies apart into
        runaway spin. A fixed seed keeps the whole scatter reproducible, so the
        golden-master oracle settles to a stable recorded state.
    """
    rng = random.Random(seed)
    dynamics = []
    placed = []
    while len(dynamics) < 24:
        angle = rng.uniform(0, 6.28)
        kind = rng.random()
        if kind < 0.4:
            radius = rng.uniform(0.6, 1.2)
            bound = radius
            spec = BodySpec("circle", (200, 100, 50), (0, 0), angle=angle, radius=radius)
        elif kind < 0.7:
            width = rng.uniform(1.2, 2.2)
            height = rng.uniform(1.2, 2.2)
            bound = 0.5 * math.hypot(width, height)
            spec = BodySpec("rectangle", (50, 120, 200), (0, 0), angle=angle,
                            width=width, height=height)
        else:
            radius = rng.uniform(0.8, 1.3)
            bound = radius
            spec = BodySpec("regular_polygon", (180, 60, 160), (0, 0), angle=angle,
                            num_sides=rng.randint(3, 6), radius=radius)
        x, y = sample_free_position(rng, placed, bound)
        dynamics.append(spec._replace(position=(x, y)))
        placed.append((x, y, bound))
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
    generator = GeneratorSpec((ShapeCategory("circle"),), x_range=(-4.0, 4.0),
                              y_range=(wall_top + 0.5, wall_top + 0.5),
                              size_range=(1.2, 1.2), rate=3.0, randomize_color=True)
    view_height = machine_height + 5.5
    return Scene("pachinko", tuple(statics), (), (generator,),
                 view_height=view_height, view_aspect=24.0 / view_height)


def _drop_emitter(x_range: Tuple[float, float], rate: float) -> GeneratorSpec:
    """A single emitter raining a mix of large convex shapes from above the arena."""
    shapes = (ShapeCategory("circle"), ShapeCategory("rectangle"),
              ShapeCategory("regular_polygon", num_sides=3),
              ShapeCategory("regular_polygon", num_sides=5),
              ShapeCategory("regular_polygon", num_sides=6))
    return GeneratorSpec(shapes, x_range=x_range, y_range=(-13.0, -13.0),
                         size_range=(1.6, 2.4), rate=rate, randomize_color=True, seed=1)


def make_default_drop_scene() -> Scene:
    """The default ramp scene with an emitter raining shapes onto its ledges."""
    return Scene("default_drop", DEFAULT_SCENE.statics, (), (_drop_emitter((-8.0, 8.0), 6.0),))


def make_open_box_drop_scene() -> Scene:
    """The open box steadily filled by an emitter raining shapes into it."""
    return Scene("open_box_drop", OPEN_BOX.statics, (), (_drop_emitter((-9.0, 9.0), 10.0),))


BUILTIN_SCENES = {scene.name: scene for scene in (
    DEFAULT_SCENE, OPEN_BOX, make_stack_scene(), make_pyramid_scene(),
    make_golden_scene(), make_pachinko_scene(),
    make_default_drop_scene(), make_open_box_drop_scene())}
