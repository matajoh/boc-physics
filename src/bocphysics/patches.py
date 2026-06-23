"""Loose-quadtree spatial partition that cuts the world into independent patches.

Description:
    The broad phase yields a flat list of candidate colliding pairs. To solve
    those pairs in parallel we must first decide which worker owns which pair.
    This module assigns every dynamic body to exactly one patch -- the loose
    quadtree cell its centre falls in -- and then classifies every candidate
    pair. A pair whose two bodies share a patch is interior to that patch; a
    pair spanning two patches is a boundary pair linking the two patches; a
    dynamic-static contact is assigned to the dynamic body's patch and the
    static's geometry reaches that worker through the shared geometry snapshot
    so it can resolve it without reaching across a seam. The patches tile the
    world and exchange only a thin halo of replicated statics, mirroring the
    domain-decomposition "patch + halo" pattern used in parallel physics
    solvers, and
    the name patch is used deliberately rather than region, which has a
    distinct meaning in the Behavior-Oriented Concurrency literature.
"""

from typing import Dict, List, Tuple

from .bodies import AABB, RigidBody
from .quadtree import LooseQuadTree

Pair = Tuple[RigidBody, RigidBody]


class Patch:
    """The dynamic bodies in one loose-quadtree cell plus the work local to them."""

    def __init__(self):
        """Create an empty patch with no bodies or interior pairs."""
        self.bodies: List[RigidBody] = []
        self.interior_pairs: List[Pair] = []


class BoundaryPair:
    """A candidate pair whose two dynamic bodies live in different patches."""

    def __init__(self, patch_a: int, patch_b: int, pair: Pair):
        """Record the two patch indices and the pair."""
        self.patch_a = patch_a
        self.patch_b = patch_b
        self.pair = pair


class Partition:
    """The patches the world was cut into and the boundary pairs that stitch them."""

    def __init__(self, patches: List[Patch], boundary_pairs: List[BoundaryPair]):
        """Create a partition from its patches and boundary pairs."""
        self.patches = patches
        self.boundary_pairs = boundary_pairs


def route_pairs(patches: List[Patch], patch_of: Dict[int, int],
                collisions: List[Pair]) -> Partition:
    """Classify every candidate pair into the given patches by body->patch map.

    Description:
        The shared tail of every partition strategy: only how `patch_of` is
        computed differs (quadtree cell vs slab bin), the routing is identical.
        Dynamic-dynamic pairs become interior or boundary work, dynamic-static
        contacts attach to the dynamic body's patch, and static-static pairs are
        dropped, exactly as the serial island builder drops them. Endpoint order
        is preserved throughout so a contact normal is never flipped.
    """
    boundary_pairs: List[BoundaryPair] = []
    for a, b in collisions:
        if a.physics and b.physics:
            patch_a = patch_of[a.uid]
            patch_b = patch_of[b.uid]
            if patch_a == patch_b:
                patches[patch_a].interior_pairs.append((a, b))
            else:
                boundary_pairs.append(BoundaryPair(patch_a, patch_b, (a, b)))
        elif a.physics or b.physics:
            dynamic = a if a.physics else b
            patches[patch_of[dynamic.uid]].interior_pairs.append((a, b))
        # both static: dropped, matching the serial island builder

    return Partition(patches, boundary_pairs)


def build_partition(bodies: List[RigidBody], collisions: List[Pair], box: AABB,
                    coarsen: float = 2.0, threshold: int = 8) -> Partition:
    """Cut the world into patches and classify every candidate pair.

    Description:
        Dynamic bodies are inserted into a centre-based loose quadtree; each
        cell holding bodies becomes a patch. Every candidate pair is then routed
        by route_pairs: dynamic-dynamic pairs become interior or boundary work,
        and dynamic-static contacts attach to the dynamic body's patch, the
        static reaching that worker through the shared geometry snapshot.
    """
    tree = LooseQuadTree(box, coarsen=coarsen, threshold=threshold)
    for body in bodies:
        if body.physics:
            tree.insert(body)

    patch_nodes = tree.cells()
    patches = [Patch() for _ in patch_nodes]
    patch_of: Dict[int, int] = {}
    for index, node in enumerate(patch_nodes):
        for body in node.values:
            patch_of[body.uid] = index
            patches[index].bodies.append(body)

    return route_pairs(patches, patch_of, collisions)


def build_slab_partition(bodies: List[RigidBody], collisions: List[Pair],
                         box: AABB, num_slabs: int, axis: str = "x",
                         min_slab_bodies: int = 1) -> Partition:
    """Cut the world into equal-population slabs along one axis, then route pairs.

    Description:
        Dynamic bodies are sorted along the axis and split into num_slabs bins of
        equal population, so each patch owns the same body count regardless of how
        the pile is shaped. A vertical (x-axis) cut severs few stacked contacts
        under gravity, keeping the seam graph shallow -- far shallower than the
        loose-quadtree cut -- which is what bounds the parallel barrier depth. The
        box is unused (slabs are data-defined, not world-defined) but kept in the
        signature so the two builders are interchangeable. With no dynamic bodies
        the partition is empty; a single slab yields one patch and no seams.

        min_slab_bodies floors each slab's population: the slab count is capped at
        total // min_slab_bodies, so a small or sparse scene collapses to fewer,
        fuller slabs instead of degenerate one-body slabs that would turn every
        interior pair into a seam. With the default of 1 the cap is inert.
    """
    dynamic = [body for body in bodies if body.physics]
    coord = (lambda body: body.position.x) if axis == "x" else (lambda body: body.position.y)
    dynamic.sort(key=coord)
    total = len(dynamic)
    population_cap = total // max(1, min_slab_bodies)
    count = max(1, min(num_slabs, population_cap)) if total else 0
    patches = [Patch() for _ in range(count)]
    patch_of: Dict[int, int] = {}
    for rank, body in enumerate(dynamic):
        index = min(count - 1, rank * count // total)
        patch_of[body.uid] = index
        patches[index].bodies.append(body)

    return route_pairs(patches, patch_of, collisions)
