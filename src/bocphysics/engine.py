"""Module containing the physics engine."""

from typing import List, Set, Tuple

from bocpy import Matrix
from pyglet import shapes
from pyglet.graphics import Batch

from .bodies import AABB, RigidBody
from .collisions import detect_collision
from .config import DetectionKind, PhysicsMode
from .contacts import find_contact_points
from .detection import Detection
from .physics import Physics
from .render import BLACK, Camera, YELLOW


ZERO_VEC = Matrix.vector([0, 0])


class Island:
    """A connected group of dynamic bodies solved independently each frame.

    Description:
        Islands partition the world so that disjoint groups of interacting
        bodies can be resolved without affecting one another. A singleton
        island holds one dynamic body with no candidate contacts.
    """

    def __init__(self):
        """Create an empty island with no bodies or candidate pairs."""
        self.bodies: List[RigidBody] = []
        self.pairs: List[Tuple[RigidBody, RigidBody]] = []


class PhysicsEngine:
    """Class representing the physics engine.

    Description:
        The physics engine is in charge of maintaining the list of
        rigid bodies and causing them to interact with each other, while
        managing the state of the simulation.

    Args:
        width (float): The width (in pixels) of the simulation window
        height (float): The height (in pixels) of the simulation window
        mode (PhysicsMode): The physics mode to use
        detection_kind (DetectionKind): The collision detection algorithm to use
        show_contacts (bool): Whether to display contact points
        height_in_meters (float): The height of the simulation window in meters
    """

    def __init__(self, width: float, height: float,
                 mode: PhysicsMode, detection_kind: DetectionKind,
                 show_contacts: bool, height_in_meters=30):
        """Create the engine from the window size, physics mode, and detection kind."""
        self.scale = height / height_in_meters
        self.width = width / self.scale
        self.height = height_in_meters
        self.physics = Physics(mode)
        self.bounds = AABB(-self.width / 2, -self.width / 2,
                           self.width / 2, self.height / 2)
        self.detection = Detection(detection_kind, AABB(-self.width, -self.height,
                                                        self.width, self.height))
        self.bodies: List[RigidBody] = []
        self.gravity = Matrix.vector([0, 9.81])
        self.collisions: List[Tuple[RigidBody, RigidBody]] = []
        self.to_remove: List[RigidBody] = []
        self.center = Matrix.vector([self.width / 2, self.height / 2])
        self.contacts: Set[Tuple[float, float]] = set()
        self.show_contacts = show_contacts
        self.mode = mode
        # the swept AABB pads each body's box to absorb a frame's motion
        self.swept_slop = 0.25
        # the systems are defined by which components they operate over
        self.systems = {
            "physics": ["position", "angle",
                        "linear_velocity", "angular_velocity",
                        "mass", "inertia"],
            "collision": ["aabb"],
            "render": ["draw"]
        }

    def remove_outside(self):
        """Removes bodies that are outside the bounds of the simulation."""
        self.to_remove.clear()
        for body in self.bodies:
            if body.aabb.disjoint(self.bounds):
                # if the body is outside the bounds of the simulation,
                # we can safely remove it
                self.to_remove.append(body)

        for body in self.to_remove:
            self.bodies.remove(body)

    def broad_phase(self):
        """Performs the broad phase of collision detection.

        Description:
            This phase focuses on determining quickly which pairs of bodies
            are potentially colliding. It does this by checking if the
            bounding boxes of the bodies intersect.
        """
        return self.detection.find_all_intersections(self.bodies, self.collisions)

    def update_swept_aabbs(self, dt: float):
        """Compute each body's swept AABB for this frame's broad phase.

        Description:
            The swept box grows a body's tight AABB along its frame motion so
            a single broad-phase pass yields candidate pairs valid for every
            sub-step. Statics do not move, so they grow only by the slop.
        """
        world = self.detection.box
        for body in self.bodies:
            displacement = body.linear_velocity * dt if body.physics else ZERO_VEC
            swept = body.aabb.sweep(displacement, self.swept_slop)
            # clamp to the detection world so the quadtree always contains it
            body.swept_aabb = AABB(max(swept.left, world.left), max(swept.top, world.top),
                                   min(swept.right, world.right), min(swept.bottom, world.bottom))

    def build_islands(self) -> List[Island]:
        """Partition dynamic bodies into islands via union-find over pairs.

        Description:
            Two dynamic bodies sharing a candidate pair join the same island.
            Statics carry no island membership; a dynamic-static pair attaches
            to the dynamic body's island. Static-static pairs are dropped.
        """
        parent = {}

        def find(body):
            while parent[body] is not body:
                parent[body] = parent[parent[body]]
                body = parent[body]

            return body

        for body in self.bodies:
            if body.physics:
                parent[body] = body

        for a, b in self.collisions:
            if a.physics and b.physics:
                parent[find(a)] = find(b)

        islands = {}
        for body in parent:
            islands.setdefault(find(body), Island()).bodies.append(body)

        for a, b in self.collisions:
            if a.physics:
                islands[find(a)].pairs.append((a, b))
            elif b.physics:
                islands[find(b)].pairs.append((a, b))

        return list(islands.values())

    def resolve_pair(self, a: RigidBody, b: RigidBody):
        """Detect and resolve a contact between one candidate pair of bodies.

        Description:
            This is the narrow phase for a single pair: it confirms the exact
            collision, generates contact points, and passes them to the
            physics module for impulse resolution.
        """
        collision = detect_collision(a, b)
        if collision is None:
            # false positive
            return

        c0, c1 = find_contact_points(a, b, collision)
        if self.show_contacts:
            self.contacts.add((c0.x, c0.y))
            if c1 is not None:
                self.contacts.add((c1.x, c1.y))

        self.physics.resolve_collision(a, b, collision, c0, c1)

    def solve_island(self, island: Island, dt: float, num_iterations: int):
        """Advance one island over all sub-steps, integrating then resolving."""
        for _ in range(num_iterations):
            for body in island.bodies:
                body.step(dt, self.gravity)

            for a, b in island.pairs:
                self.resolve_pair(a, b)

    def step(self, dt: float, num_iterations=20):
        """Advances the simulation by a time step.

        Args:
            dt (float): The time step to advance the simulation by
            num_iterations (int): The number of iterations to perform
                                  in each time step. More iterations results
                                  in more accurate physics, but is more
                                  computationally expensive.
        """
        # the main problem inherent in a posteriori physics simulation
        # is "tunneling" where objects pass through each other.
        # to mitigate this, we subdivide the time step into smaller
        # increments and resolve collisions at each step using impulses.
        self.contacts.clear()
        self.update_swept_aabbs(dt)

        # broad phase and island partition are built once per frame, then
        # each island runs all sub-steps over its own bodies and pairs.
        self.collisions.clear()
        self.broad_phase()
        islands = self.build_islands()

        sub_dt = dt / num_iterations
        for island in islands:
            self.solve_island(island, sub_dt, num_iterations)

        self.remove_outside()

    def draw(self, batch: Batch, project: Camera) -> list:
        """Draws the simulation into the batch, returning shapes to keep alive."""
        kept = []
        for body in self.bodies:
            if body.render:
                kept.extend(body.draw(batch, project))

        for contact in self.contacts:
            x, y = project(Matrix.vector([contact[0], contact[1]]))
            kept.append(shapes.Circle(x, y, 5, color=YELLOW, batch=batch))
            kept.append(shapes.Arc(x, y, 5, thickness=2, color=BLACK, batch=batch))

        return kept

    def add_body(self, body: RigidBody):
        """Adds a body to the simulation."""
        for system, components in self.systems.items():
            # we will store a boolean for each system indicating
            # whether the body has all the necessary components
            has_system = all(hasattr(body, component) for component in components)
            setattr(body, system, has_system)

        self.bodies.append(body)

    def remove_body(self, body: RigidBody):
        """Removes a body from the simulation."""
        self.bodies.remove(body)

    def to_world(self, pos: Matrix) -> Matrix:
        """Converts a position from screen coordinates to world coordinates.

        Args:
            pos (Matrix): The position to convert

        Returns:
            Matrix: The position in world coordinates
        """
        return pos / self.scale - self.center
