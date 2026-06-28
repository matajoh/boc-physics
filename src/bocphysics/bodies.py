"""Module providing basic circle and polygon bodies."""

import math
from typing import List, NamedTuple, Tuple, Union

from bocpy import Matrix


Color = Tuple[int, int, int]


class AABB(NamedTuple("AABB", [("left", float), ("top", float), ("right", float), ("bottom", float)])):
    """An axis-aligned bounding box."""

    def disjoint(self, other: "AABB") -> bool:
        """Check if this AABB is disjoint from another AABB.

        Description:
            Note how this test is designed such that it never needs to
            do more than the minimum number of comparisons.
        """
        return (self.left > other.right or
                self.right < other.left or
                self.top > other.bottom or
                self.bottom < other.top)

    def contains(self, other: "AABB") -> bool:
        """Check if this AABB contains another AABB."""
        return not (self.left > other.left or
                    self.right < other.right or
                    self.top > other.top or
                    self.bottom < other.bottom)

    def intersects(self, other: "AABB") -> bool:
        """Check if this AABB intersects with another AABB.

        Description:
            Using the negation of disjoint means that we can avoid
            unnecessary tests. This is an example of De Morgan's law.
        """
        return not self.disjoint(other)

    @property
    def top_left(self) -> Matrix:
        """Get the top left corner of the AABB."""
        return Matrix.vector([self.left, self.top])

    @property
    def center(self) -> Matrix:
        """Get the center of the AABB."""
        return Matrix.vector([(self.left + self.right) / 2, (self.top + self.bottom) / 2])

    @property
    def size(self) -> Matrix:
        """Get the size of the AABB."""
        return Matrix.vector([self.right - self.left, self.bottom - self.top])

    def sweep(self, displacement: Matrix, slop: float) -> "AABB":
        """Return this box grown to cover motion by displacement, padded by slop."""
        return AABB(min(self.left, self.left + displacement.x) - slop,
                    min(self.top, self.top + displacement.y) - slop,
                    max(self.right, self.right + displacement.x) + slop,
                    max(self.bottom, self.bottom + displacement.y) + slop)

    @staticmethod
    def create(top_left: Matrix, size: Matrix) -> "AABB":
        """Create an AABB from a top-left corner and a size."""
        return AABB(top_left.x, top_left.y, top_left.x + size.x, top_left.y + size.y)


class Circle:
    """A circle body."""

    def __init__(self,
                 radius: float,
                 color: Color,
                 linear_velocity: Matrix = None,
                 angular_velocity: float = None,
                 mass: float = float("inf"),
                 inv_mass: float = 0,
                 inertia: float = float("inf"),
                 inv_inertia: float = 0):
        """Create a circle from its radius, colour, and mass properties."""
        self.position = Matrix.vector([0, 0])
        self.angle = 0
        self.uid = None
        self.radius = radius
        self.color = color
        self.mass = mass
        self.inv_mass = inv_mass
        self.inertia = inertia
        self.inv_inertia = inv_inertia
        if linear_velocity is not None:
            self.linear_velocity = linear_velocity
            self.angular_velocity = angular_velocity

        self.size = radius * 2
        self.aabb_ = AABB(0, 0, 0, 0)
        self.swept_aabb = self.aabb_
        self.update_needed_ = True

    def step(self, dt: float, gravity: Matrix):
        """Integrate the circle's velocity and position over the time step."""
        self.linear_velocity = self.linear_velocity + gravity * dt
        self.position = self.position + self.linear_velocity * dt
        self.angle = self.angle + self.angular_velocity * dt
        self.update_needed_ = True

    def move_to(self, pos: Matrix) -> "Circle":
        """Move the circle to an absolute position and return it."""
        self.position = pos.copy()
        self.update_needed_ = True
        return self

    def move(self, delta: Matrix) -> "Circle":
        """Move the circle by a relative delta and return it."""
        self.position = self.position + delta
        self.update_needed_ = True
        return self

    def rotate_to(self, angle: float) -> "Circle":
        """Rotate the circle to an absolute angle and return it."""
        self.angle = angle
        self.update_needed_ = True
        return self

    def to_dict(self):
        """Return a JSON-serialisable dict describing the circle."""
        return {"kind": "circle",
                "position": (self.position.x, self.position.y),
                "angle": self.angle,
                "radius": self.radius,
                "color": self.color}

    @property
    def aabb(self) -> AABB:
        """Get the axis-aligned bounding box of the circle.

        Description:
            Note how the bounding box is only updated when necessary.
        """
        if self.update_needed_:
            self.aabb_ = AABB(self.position.x - self.radius, self.position.y - self.radius,
                              self.position.x + self.radius, self.position.y + self.radius)
            self.update_needed_ = False

        return self.aabb_

    @staticmethod
    def create(radius: float, density: float, color: Color, is_static=False):
        """Create a circle, computing its mass and inertia from density."""
        if is_static:
            return Circle(radius, color)

        mass = radius**2 * math.pi * density
        inv_mass = 1 / mass
        inertia = 0.5 * mass * radius * radius
        inv_inertia = 1 / inertia
        return Circle(radius, color,
                      Matrix.vector([0, 0]), 0,
                      mass, inv_mass,
                      inertia, inv_inertia)


class Polygon:
    """A polygon body.

    Description:
        Polygons can be either static or dynamic. As such, the
        constructor provides default values for static polygons
        which will ensure they interact properly with dynamic objects.
    """

    def __init__(self,
                 vertices: List[Matrix],
                 normals: List[Matrix],
                 color: Color,
                 linear_velocity: Matrix = None,
                 angular_velocity: float = None,
                 mass: float = float("inf"),
                 inv_mass: float = 0,
                 inertia: float = float("inf"),
                 inv_inertia: float = 0):
        """Create a polygon from its vertices, normals, colour, and mass properties."""
        self.position = Matrix.vector([0, 0])
        self.angle = 0
        self.uid = None
        self.vertices = vertices
        self.normals = normals
        self.radius = max(v.length for v in vertices)
        self.color = color
        self.mass = mass
        self.inv_mass = inv_mass
        self.inertia = inertia
        self.inv_inertia = inv_inertia
        if linear_velocity is not None:
            self.linear_velocity = linear_velocity
            self.angular_velocity = angular_velocity

        self.aabb_ = AABB(0, 0, 0, 0)
        self.swept_aabb = self.aabb_
        self.vertices_block_ = Matrix(len(vertices), 2, [c for v in vertices for c in (v.x, v.y)])
        self.normals_block_ = Matrix(len(normals), 2, [c for n in normals for c in (n.x, n.y)])
        self.transformed_vertices_block_ = self.vertices_block_.copy()
        self.transformed_normals_block_ = self.normals_block_.copy()
        self.update_needed_ = True

    def step(self, dt: float, gravity: Matrix):
        """Integrate the polygon's velocity and position over the time step."""
        self.linear_velocity = self.linear_velocity + gravity * dt
        self.position = self.position + self.linear_velocity * dt
        self.angle = self.angle + self.angular_velocity * dt
        self.update_needed_ = True

    def move_to(self, pos: Matrix) -> "Polygon":
        """Move the polygon to an absolute position and return it."""
        self.position = pos.copy()
        self.update_needed_ = True
        return self

    def move(self, delta: Matrix) -> "Polygon":
        """Move the polygon by a relative delta and return it."""
        self.position = self.position + delta
        self.update_needed_ = True
        return self

    def rotate_to(self, angle: float) -> "Polygon":
        """Rotate the polygon to an absolute angle and return it."""
        self.angle = angle
        self.update_needed_ = True
        return self

    def update_transform(self):
        """Update the transformed vertices and normals.

        Description:
            We want to avoid updating the transformed vertices and normals
            unless absolutely necessary. This is an example of lazy
            evaluation using a "dirty" bit, in this case the ``update_needed_``
            flag. The rotation is a single batched matrix product over the
            whole (N x 2) vertex block rather than a per-vertex Python loop.
        """
        if not self.update_needed_:
            return

        self.update_needed_ = False
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        rot_t = Matrix(2, 2, [cos_angle, sin_angle, -sin_angle, cos_angle])
        self.transformed_normals_block_ = self.normals_block_ @ rot_t
        self.transformed_vertices_block_ = self.vertices_block_ @ rot_t + self.position

        low = self.transformed_vertices_block_.min(axis=0)
        high = self.transformed_vertices_block_.max(axis=0)
        self.aabb_ = AABB(low.x, low.y, high.x, high.y)

    def to_dict(self):
        """Return a JSON-serialisable dict describing the polygon."""
        return {"kind": "polygon",
                "position": (self.position.x, self.position.y),
                "angle": self.angle,
                "vertices": [(v.x, v.y) for v in self.vertices],
                "color": self.color}

    @property
    def aabb(self) -> AABB:
        """Get the axis-aligned bounding box of the polygon."""
        self.update_transform()
        return self.aabb_

    @property
    def transformed_vertices(self) -> Matrix:
        """Get the polygon's vertices in world space as an (N x 2) block."""
        self.update_transform()
        return self.transformed_vertices_block_

    @property
    def transformed_normals(self) -> Matrix:
        """Get the polygon's edge normals in world space as an (N x 2) block."""
        self.update_transform()
        return self.transformed_normals_block_

    @staticmethod
    def create_rectangle(width: float, height: float, density: float,
                         color: Color, is_static=False):
        """Create a rectangular polygon from its width, height, and density."""
        vertices = [Matrix.vector([-width / 2, -height / 2]),
                    Matrix.vector([width / 2, -height / 2]),
                    Matrix.vector([width / 2, height / 2]),
                    Matrix.vector([-width / 2, height / 2])]
        normals = [Matrix.vector([0, 1]), Matrix.vector([1, 0])]
        if is_static:
            return Polygon(vertices, normals, color)

        mass = width * height * density
        inv_mass = 1 / mass
        inertia = (1 / 12) * mass * (width**2 + height**2)
        inv_inertia = 1 / inertia

        return Polygon(vertices, normals, color,
                       Matrix.vector([0, 0]), 0,
                       mass, inv_mass,
                       inertia, inv_inertia)

    @staticmethod
    def create_regular_polygon(num_sides: int, radius: float, density: float,
                               color: Color, is_static=False):
        """Create a regular polygon from its side count, radius, and density."""
        angle = 2 * math.pi / num_sides
        vertices = [Matrix.vector([radius * math.cos(i * angle), radius * math.sin(i * angle)])
                    for i in range(num_sides)]
        normals = []
        # even polygons have parallel opposite edges, so only half the face normals are distinct SAT axes
        if num_sides % 2 == 0:
            for i in range(num_sides // 2):
                n = (vertices[i] + vertices[i + 1]).normalize()
                normals.append(n)
        else:
            for i in range(num_sides):
                n = (vertices[i] + vertices[(i + 1) % num_sides]).normalize()
                normals.append(n)

        if is_static:
            return Polygon(vertices, normals, color)

        apothem = radius * math.cos(angle / 2)
        perimeter = num_sides * (vertices[1] - vertices[0]).length
        mass = 0.5 * perimeter * apothem * density
        inv_mass = 1 / mass
        # moment of inertia of a regular n-gon about its centroid
        inertia = 0.5 * mass * radius**2 * (1 - 2 / 3 * math.sin(math.pi / num_sides)**2)
        inv_inertia = 1 / inertia

        return Polygon(vertices, normals, color,
                       Matrix.vector([0, 0]), 0,
                       mass, inv_mass,
                       inertia, inv_inertia)


RigidBody = Union[Circle, Polygon]
