"""Module providing broad-phase collision detection strategies."""

from typing import List, Tuple

from .bodies import AABB, RigidBody
from .config import DetectionKind
from .patches import build_partition
from .quadtree import QuadTree


Collisions = List[Tuple[RigidBody, RigidBody]]


class Detection:
    """Selects and runs the configured broad-phase detection algorithm."""

    def __init__(self, kind: DetectionKind, box: AABB):
        """Create a detector from its algorithm kind and world bounds."""
        self.kind = kind
        self.box = box

    def find_all_intersections(self, bodies: List[RigidBody], collisions: Collisions):
        """Find all colliding pairs using the configured algorithm."""
        match self.kind:
            case DetectionKind.QUADTREE:
                return self.find_all_intersections_quadtree(bodies, collisions)
            case DetectionKind.LOOSE_QUADTREE:
                return self.find_all_intersections_loose_quadtree(bodies, collisions)
            case DetectionKind.BASIC:
                return self.find_all_intersections_basic(bodies, collisions)
            case _:
                raise ValueError("Invalid detection kind")

    def find_all_intersections_quadtree(self, bodies: List[RigidBody], collisions: Collisions):
        """Find all colliding pairs using a quadtree spatial index."""
        quadtree = QuadTree(self.box)
        for body in bodies:
            if body.collision:
                quadtree.add(body)

        quadtree.find_all_intersections(collisions)

    def find_all_intersections_loose_quadtree(self, bodies: List[RigidBody],
                                              collisions: Collisions):
        """Find colliding pairs, then route them through the loose partition.

        Description:
            The candidate pairs come from the quadtree so none are missed, but
            they are then classified into patches and re-emitted as interior,
            boundary, and dynamic-static work. This exercises the partition the
            parallel solver depends on while keeping the serial path selectable
            and testable without any concurrency. Static-static pairs, which
            resolve to a no-op, are dropped by the partition.
        """
        candidates: Collisions = []
        self.find_all_intersections_quadtree(bodies, candidates)
        partition = build_partition(bodies, candidates, self.box)
        for patch in partition.patches:
            collisions.extend(patch.interior_pairs)

        for boundary in partition.boundary_pairs:
            collisions.append(boundary.pair)

    def find_all_intersections_basic(self, bodies: List[RigidBody], collisions: Collisions):
        """Find all colliding pairs with a brute-force pairwise scan."""
        for i in range(0, len(bodies)):
            a = bodies[i]
            if not a.collision:
                continue

            for j in range(0, i):
                b = bodies[j]
                if not b.collision:
                    continue

                if a.swept_aabb.intersects(b.swept_aabb):
                    collisions.append((a, b))
