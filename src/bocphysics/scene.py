"""Module providing declarative static-scene specifications.

A scene is a small, picklable description of the static bodies that make
up the fixed parts of a simulation (floors, walls, ledges). Keeping the
scene as data rather than hard-wired construction code lets the same
engine run different arenas -- the default interactive scene, a benchmark
box -- and lets scenes be loaded from a file (see :meth:`Scene.from_dict`).
"""

import json
import math
from typing import List, NamedTuple, Tuple

from bocpy import Matrix

from .bodies import Circle, Polygon, RigidBody
from .render import Color


class StaticBody(NamedTuple):
    """A declarative spec for one static body in a scene."""

    kind: str
    color: Color
    position: Tuple[float, float]
    angle: float = 0.0
    width: float = 0.0
    height: float = 0.0
    num_sides: int = 0
    radius: float = 0.0
    density: float = 2.0

    def build(self) -> RigidBody:
        """Construct the concrete static rigid body from this spec."""
        if self.kind == "rectangle":
            body = Polygon.create_rectangle(self.width, self.height,
                                            self.density, self.color, is_static=True)
        elif self.kind == "regular_polygon":
            body = Polygon.create_regular_polygon(self.num_sides, self.radius,
                                                  self.density, self.color, is_static=True)
        elif self.kind == "circle":
            body = Circle.create(self.radius, self.density, self.color, is_static=True)
        else:
            raise ValueError(f"Unknown static body kind: {self.kind}")

        body.move_to(Matrix.vector([self.position[0], self.position[1]]))
        body.rotate_to(self.angle)
        return body

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict describing this static body."""
        return self._asdict()

    @staticmethod
    def from_dict(data: dict) -> "StaticBody":
        """Create a static body spec from a dict, applying field defaults."""
        fields = {key: data[key] for key in StaticBody._fields if key in data}
        if "position" in fields:
            fields["position"] = tuple(fields["position"])
        if "color" in fields and isinstance(fields["color"], list):
            fields["color"] = tuple(fields["color"])
        return StaticBody(**fields)


class Scene(NamedTuple):
    """A named collection of static bodies."""

    name: str
    statics: Tuple[StaticBody, ...]

    def build(self) -> List[RigidBody]:
        """Construct all of the scene's static rigid bodies."""
        return [spec.build() for spec in self.statics]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict describing this scene."""
        return {"name": self.name,
                "statics": [spec.to_dict() for spec in self.statics]}

    @staticmethod
    def from_dict(data: dict) -> "Scene":
        """Create a scene from a dict produced by :meth:`to_dict`."""
        statics = tuple(StaticBody.from_dict(spec) for spec in data["statics"])
        return Scene(data["name"], statics)

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
    StaticBody("rectangle", "darkgreen", (0, 10), width=30, height=2),
    StaticBody("rectangle", "gray", (-7.5, 0), angle=math.pi / 10, width=15, height=1),
    StaticBody("rectangle", "darkred", (7, -5), angle=-math.pi / 10, width=10, height=1),
))

# An open-topped box for dropping shapes into during benchmarks.
OPEN_BOX = Scene("open_box", (
    StaticBody("rectangle", "darkgreen", (0, 10), width=30, height=2),
    StaticBody("rectangle", "gray", (-14, -2), width=2, height=24),
    StaticBody("rectangle", "gray", (14, -2), width=2, height=24),
))

# Built-in scenes addressable by name on the command line.
BUILTIN_SCENES = {scene.name: scene for scene in (DEFAULT_SCENE, OPEN_BOX)}
