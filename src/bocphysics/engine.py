"""Module containing the physics engine."""

from typing import List, Set, Tuple

from bocpy import Matrix

from . import solver, xpbd, xpbd_kernel
from .bodies import AABB, RigidBody
from .config import DetectionKind, PhysicsMode
from .detection import Detection
from .physics import Physics


ZERO_VEC = Matrix.vector([0, 0])


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
        num_substeps (int): Sub-steps per frame for the substep solver
        num_velocity_iterations (int): Velocity iterations per sub-step for the
                                       substep solver
    """

    def __init__(self, width: float, height: float,
                 mode: PhysicsMode, detection_kind: DetectionKind,
                 show_contacts: bool, height_in_meters=30,
                 num_substeps=4, num_velocity_iterations=10):
        """Create the engine from the window size, physics mode, and detection kind."""
        self.scale = height / height_in_meters
        self.width = width / self.scale
        self.height = height_in_meters
        self.physics = Physics(mode)
        self.bounds = AABB(-self.width / 2, -self.height / 2,
                           self.width / 2, self.height / 2)
        self.detection = Detection(detection_kind, AABB(-self.width, -self.height,
                                                        self.width, self.height))
        self.bodies: List[RigidBody] = []
        self.gravity = Matrix.vector([0, 9.81])
        self.collisions: List[Tuple[RigidBody, RigidBody]] = []
        self.to_remove: List[RigidBody] = []
        self.next_uid = 0
        self.center = Matrix.vector([self.width / 2, self.height / 2])
        self.contacts: Set[Tuple[float, float]] = set()
        self.show_contacts = show_contacts
        self.mode = mode
        self.num_substeps = num_substeps
        self.num_velocity_iterations = num_velocity_iterations
        self.swept_slop = 0.25
        self.systems = {
            "physics": ["position", "angle",
                        "linear_velocity", "angular_velocity",
                        "mass", "inertia"],
            "collision": ["aabb"],
            "render": ["position", "color"]
        }

    def remove_outside(self):
        """Removes moving bodies that have drifted outside the simulation bounds."""
        self.to_remove.clear()
        for body in self.bodies:
            if body.physics and body.aabb.disjoint(self.bounds):
                self.to_remove.append(body)

        for body in self.to_remove:
            self.bodies.remove(body)

    def broad_phase(self):
        """Performs the broad phase of collision detection.

        Description:
            Quickly finds which pairs of bodies might be colliding by
            testing whether their bounding boxes intersect.
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
            body.swept_aabb = AABB(max(swept.left, world.left), max(swept.top, world.top),
                                   min(swept.right, world.right), min(swept.bottom, world.bottom))

    def solve_substep(self, bodies: List[RigidBody],
                      pairs: List[Tuple[RigidBody, RigidBody]], sub_dt: float):
        """Advance every dynamic body, separating sub-steps from velocity iterations.

        Description:
            Each sub-step integrates the bodies then builds every pair's
            contact manifold once, since the geometry barely moves within a
            sub-step. The velocity solver then iterates over those cached
            manifolds, converging the coupled contacts without paying the
            narrow-phase cost again. The work is delegated to the shared
            solver core so the parallel path runs the identical solve.
        """
        contacts = self.contacts if self.show_contacts else None
        if solver.use_batched_solver:
            xpbd_kernel.solve_group_substep(self.physics, bodies, pairs,
                                            self.gravity, sub_dt, self.num_substeps,
                                            contacts)
        else:
            xpbd.solve_group_substep(self.physics, bodies, pairs,
                                     self.gravity, sub_dt, self.num_substeps, contacts)

    def step(self, dt: float):
        """Advances the simulation by a time step.

        Args:
            dt (float): The time step to advance the simulation by
        """
        self.contacts.clear()
        self.update_swept_aabbs(dt)

        self.collisions.clear()
        self.broad_phase()
        bodies = [body for body in self.bodies if body.physics]
        pairs = [(a, b) for a, b in self.collisions if a.physics or b.physics]

        sub_dt = dt / self.num_substeps
        self.solve_substep(bodies, pairs, sub_dt)

        self.remove_outside()

    def add_body(self, body: RigidBody):
        """Adds a body to the simulation."""
        # a body joins a system only if it has all that system's components
        for system, components in self.systems.items():
            has_system = all(hasattr(body, component) for component in components)
            setattr(body, system, has_system)

        body.uid = self.next_uid
        self.next_uid += 1
        self.bodies.append(body)

    def to_world(self, pos: Matrix) -> Matrix:
        """Converts a position from screen coordinates to world coordinates.

        Args:
            pos (Matrix): The position to convert

        Returns:
            Matrix: The position in world coordinates
        """
        return pos / self.scale - self.center
