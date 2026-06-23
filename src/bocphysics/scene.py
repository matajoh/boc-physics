"""Module providing declarative static-scene specifications.

A scene is a small, picklable description of the static bodies that make
up the fixed parts of a simulation (floors, walls, ledges). Keeping the
scene as data rather than hard-wired construction code lets the same
engine run different arenas -- the default interactive scene, a benchmark
box -- and lets scenes be loaded from a file (see :meth:`Scene.from_dict`).
"""

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


class Scene(NamedTuple):
    """A named collection of static and dynamic bodies."""

    name: str
    statics: Tuple[BodySpec, ...]
    dynamics: Tuple[BodySpec, ...] = ()

    def build(self) -> List[RigidBody]:
        """Construct all of the scene's bodies, statics first then dynamics."""
        return ([spec.build(is_static=True) for spec in self.statics] +
                [spec.build(is_static=False) for spec in self.dynamics])

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict describing this scene."""
        return {"name": self.name,
                "statics": [spec.to_dict() for spec in self.statics],
                "dynamics": [spec.to_dict() for spec in self.dynamics]}

    @staticmethod
    def from_dict(data: dict) -> "Scene":
        """Create a scene from a dict produced by :meth:`to_dict`."""
        statics = tuple(BodySpec.from_dict(spec) for spec in data["statics"])
        dynamics = tuple(BodySpec.from_dict(spec) for spec in data.get("dynamics", ()))
        return Scene(data["name"], statics, dynamics)

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


# The default interactive scene: a floor and two angled ledges.
DEFAULT_SCENE = Scene("default", (
    BodySpec("rectangle", "darkgreen", (0, 10), width=30, height=2),
    BodySpec("rectangle", "gray", (-7.5, 0), angle=math.pi / 10, width=15, height=1),
    BodySpec("rectangle", "darkred", (7, -5), angle=-math.pi / 10, width=10, height=1),
))

# An open-topped box for dropping shapes into during benchmarks.
OPEN_BOX = Scene("open_box", (
    BodySpec("rectangle", "darkgreen", (0, 10), width=30, height=2),
    BodySpec("rectangle", "gray", (-14, -2), width=2, height=24),
    BodySpec("rectangle", "gray", (14, -2), width=2, height=24),
))

# shared layout constants for the stack and pyramid benchmark scenes
STACK_BOX_SIZE = 2.0
STACK_FLOOR_TOP = 9.0
# the seed the golden-master oracle records its settled state for
GOLDEN_SEED = 20260608


def floor_spec() -> BodySpec:
    """The wide green static floor shared by the benchmark scenes."""
    return BodySpec("rectangle", (0, 100, 0), (0, 10), width=30, height=2)


def make_stack_scene(levels: int = 8) -> Scene:
    """Build a floor with a vertical column of equal boxes resting on it."""
    dynamics = []
    for i in range(levels):
        # the world is y-down, so the lowest box sits just above the floor top
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
        # centre each row so the box above straddles two boxes below
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
    # side walls (matching OPEN_BOX) keep bodies from sliding off the floor edge
    # so the scatter settles into a contained pile instead of escaping sideways
    statics = (floor_spec(),
               BodySpec("rectangle", (120, 120, 120), (-14, -2), width=2, height=24),
               BodySpec("rectangle", (120, 120, 120), (14, -2), width=2, height=24))
    return Scene("golden", statics, tuple(dynamics))


# Built-in scenes addressable by name on the command line.
BUILTIN_SCENES = {scene.name: scene for scene in (
    DEFAULT_SCENE, OPEN_BOX, make_stack_scene(), make_pyramid_scene(),
    make_golden_scene())}
