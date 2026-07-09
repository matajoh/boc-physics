"""This module provides an implementation of a QuadTree data structure."""

from .bodies import AABB, RigidBody


class Node:
    """A node in the QuadTree."""

    def __init__(self, box: AABB):
        """Create a leaf node spanning the given box."""
        self.box = box
        self.c0 = None
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.values: list[RigidBody] = []

    def intersects(self, box: AABB) -> bool:
        """Check if this node intersects with a box."""
        return self.box.intersects(box)

    def to_leaf(self):
        """Remove the children of this node."""
        self.c0 = None
        self.c1 = None
        self.c2 = None
        self.c3 = None

    def to_inner(self):
        """Convert this node to an inner node."""
        self.c0 = Node(self.compute_box(0))
        self.c1 = Node(self.compute_box(1))
        self.c2 = Node(self.compute_box(2))
        self.c3 = Node(self.compute_box(3))

    def __getitem__(self, index: int) -> "Node":
        """Return the child node at the given quadrant index."""
        match index:
            case 0:
                return self.c0
            case 1:
                return self.c1
            case 2:
                return self.c2
            case 3:
                return self.c3
            case _:
                raise IndexError("Invalid child index")

    def __len__(self) -> int:
        """Return the fixed number of child quadrants (four)."""
        return 4

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf."""
        return self.c0 is None

    def compute_box(self, index: int) -> AABB:
        """Compute the box of a child node.

        Args:
            index: The index of the child node.
        """
        origin = self.box.top_left
        child_size = self.box.size / 2
        match index:
            case 0:
                return AABB.create(origin, child_size)
            case 1:
                return AABB.create(origin + (child_size.x, 0), child_size)
            case 2:
                return AABB.create(origin + (0, child_size.y), child_size)
            case 3:
                return AABB.create(origin + child_size, child_size)
            case _:
                raise ValueError("Invalid child index")

    def get_quadrant(self, value_box: AABB) -> int:
        """Get the quadrant of a value box in relation to a node box.

        Args:
            value_box: The box of the value.

        Returns:
            The index of the quadrant, or -1 if the value box is not in any quadrant.
        """
        center = self.box.center
        if value_box.right < center.x:
            if value_box.bottom < center.y:
                return 0
            if value_box.top >= center.y:
                return 2

        if value_box.left >= center.x:
            if value_box.bottom < center.y:
                return 1

            if value_box.top >= center.y:
                return 3

        return -1


class QuadTree:
    """A QuadTree data structure.

    Args:
        box: The box of the root node.
        max_depth: The maximum depth of the tree.
        threshold: The maximum number of values in a leaf node.
    """

    def __init__(self, box: AABB, max_depth=8, threshold=16):
        """Create a quadtree over a box with depth and leaf-size limits."""
        self.box = box
        self.root = Node(box)
        self.max_depth = max_depth
        self.threshold = threshold

    def add(self, value: RigidBody):
        """Add a value to the tree."""
        def split_(node: Node):
            assert node is not None
            assert node.is_leaf, "Only leaves can be split"
            node.to_inner()

            new_values = []
            for value in node.values:
                i = node.get_quadrant(value.swept_aabb)
                if i != -1:
                    node[i].values.append(value)
                else:
                    new_values.append(value)

            node.values = new_values

        def add_(node: Node, depth: int, value: RigidBody):
            assert node is not None
            assert node.box.contains(value.swept_aabb)
            if node.is_leaf:
                if depth >= self.max_depth or len(node.values) < self.threshold:
                    node.values.append(value)
                    return

                split_(node)

            i = node.get_quadrant(value.swept_aabb)
            if i != -1:
                add_(node[i], depth+1, value)
            else:
                node.values.append(value)

        add_(self.root, 0, value)

    def remove(self, value: RigidBody):
        """Remove a value from the tree."""
        def try_merge_(node: Node) -> bool:
            assert node is not None
            assert not node.is_leaf, "Only interior nodes can be merged"
            num_values = len(node.values)
            for child in node:
                if not child.is_leaf:
                    return False

                num_values += len(child.values)

            if num_values > self.threshold:
                return False

            for child in node:
                node.values.extend(child.values)

            node.to_leaf()
            return True

        def remove_(node: Node, value: RigidBody):
            assert node is not None
            assert node.box.contains(value.swept_aabb)
            if node.is_leaf:
                node.values.remove(value)
                return True

            i = node.get_quadrant(value.swept_aabb)
            if i != -1:
                if remove_(node[i], value):
                    return try_merge_(node)
            else:
                node.values.remove(value)

            return False

        return remove_(self.root, value)

    def query(self, box: AABB) -> list[RigidBody]:
        """Query the tree for values intersecting a box."""
        def query_(node: Node, query_box: AABB, values: list[RigidBody]):
            assert node is not None
            assert node.intersects(query_box)
            for value in node.values:
                if query_box.intersects(value.swept_aabb):
                    values.append(value)

            if node.is_leaf:
                return

            for child in node:
                if child.intersects(query_box):
                    query_(child, query_box, values)

        values = []
        query_(self.root, box, values)
        return values

    def find_all_intersections(self, intersections: list[tuple[RigidBody, RigidBody]]):
        """Find all intersections in the tree."""
        def find_intersections_in_child_(node: Node, value: RigidBody,
                                         intersections: list[tuple[RigidBody, RigidBody]]):
            if node.box.disjoint(value.swept_aabb):
                return

            for other in node.values:
                if value.swept_aabb.intersects(other.swept_aabb):
                    intersections.append((value, other))

            if node.is_leaf:
                return

            for child in node:
                find_intersections_in_child_(child, value, intersections)

        def find_all_intersections_(node: Node,
                                    intersections: list[tuple[RigidBody, RigidBody]]):
            num_values = len(node.values)
            for i in range(num_values):
                a = node.values[i]
                for j in range(0, i):
                    b = node.values[j]
                    if a.swept_aabb.intersects(b.swept_aabb):
                        intersections.append((a, b))

            if node.is_leaf:
                return

            for value in node.values:
                for child in node:
                    find_intersections_in_child_(child, value, intersections)

            for child in node:
                find_all_intersections_(child, intersections)

        find_all_intersections_(self.root, intersections)

    def boxes(self) -> list[AABB]:
        """Return the box of every node, for drawing the subdivision as an overlay."""
        found = []

        def walk(node: Node):
            found.append(node.box)
            if not node.is_leaf:
                for child in node:
                    walk(child)

        walk(self.root)
        return found
        walk(self.root)
        return found
