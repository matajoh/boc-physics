"""Immutable body geometry transport for the parallel solver.

Description:
    The (N x 7) state block carries only a patch's mutable per-substep data. To
    actually run the narrow phase and impulse solver a worker also needs each
    body's immutable geometry -- shape, mass properties, and the dynamic/static
    flag. That data changes only when bodies are spawned or removed, so it rides
    the noticeboard as a set-once snapshot the runtime caches per worker
    interpreter, rather than crossing every frame. A worker rebuilds a body
    "shell" once per uid and reuses it; the authoritative mutable state is always
    re-applied from the state block before each solve, so the shell is scratch.
"""

from typing import Dict, List, NamedTuple, Optional

from bocpy import Matrix

from .bodies import Circle, Polygon, RigidBody

Color = tuple


class BodyGeometry(NamedTuple):
    """A body's immutable shape and mass data, keyed by uid in the snapshot.

    Description:
        This is a noticeboard payload, never a cown value, so bundling Matrices
        in a NamedTuple is fine here -- the runtime pickles and caches it once
        per interpreter. position and angle are authoritative for statics and a
        harmless seed for dynamics, whose pose is overwritten from the state
        block each substep.
    """

    kind: str
    physics: bool
    color: Color
    mass: float
    inv_mass: float
    inertia: float
    inv_inertia: float
    position: Matrix
    angle: float
    radius: float
    vertices: Optional[Matrix]
    normals: Optional[Matrix]


def body_geometry(body: RigidBody) -> BodyGeometry:
    """Capture one body's immutable geometry for the noticeboard snapshot."""
    is_polygon = isinstance(body, Polygon)
    return BodyGeometry(
        kind="polygon" if is_polygon else "circle",
        physics=body.physics,
        color=body.color,
        mass=body.mass,
        inv_mass=body.inv_mass,
        inertia=body.inertia,
        inv_inertia=body.inv_inertia,
        position=body.position.copy(),
        angle=body.angle,
        radius=body.radius,
        vertices=body.vertices_block_.copy() if is_polygon else None,
        normals=body.normals_block_.copy() if is_polygon else None)


def build_geometry(bodies: List[RigidBody]) -> Dict[int, BodyGeometry]:
    """Build the uid -> BodyGeometry snapshot for a set of bodies."""
    return {body.uid: body_geometry(body) for body in bodies}


def rows_to_vectors(block: Matrix) -> List[Matrix]:
    """Split an (N x 2) block back into a list of row vectors."""
    return [Matrix.vector([block[i, 0], block[i, 1]]) for i in range(block.rows)]


def build_shell(geometry: BodyGeometry) -> RigidBody:
    """Rebuild a scratch body from its geometry, ready for state to be applied.

    Description:
        Dynamic shells are constructed with velocity components so the physics
        flag resolves true; their pose and velocity are placeholders overwritten
        from the state block. Static shells carry their authoritative pose and
        no velocity, exactly as the engine's static factories produce them.
    """
    if geometry.kind == "circle":
        if geometry.physics:
            body = Circle(geometry.radius, geometry.color, Matrix.vector([0, 0]), 0,
                          geometry.mass, geometry.inv_mass,
                          geometry.inertia, geometry.inv_inertia)
        else:
            body = Circle(geometry.radius, geometry.color)
    else:
        vertices = rows_to_vectors(geometry.vertices)
        normals = rows_to_vectors(geometry.normals)
        if geometry.physics:
            body = Polygon(vertices, normals, geometry.color, Matrix.vector([0, 0]), 0,
                           geometry.mass, geometry.inv_mass,
                           geometry.inertia, geometry.inv_inertia)
        else:
            body = Polygon(vertices, normals, geometry.color)

    body.move_to(geometry.position)
    body.rotate_to(geometry.angle)
    body.physics = geometry.physics
    body.collision = True
    body.render = True
    return body


class ShellCache:
    """Per-interpreter cache of rebuilt body shells, keyed by stable uid.

    Description:
        uids are monotonic and never reused, so a uid always maps to the same
        immutable geometry; caching the rebuilt shell avoids paying the rebuild
        on every frame. A worker keeps one instance for the life of the process,
        and evict_retired prunes shells for uids the geometry snapshot dropped so
        the cache stays bounded across spawn/remove churn.
    """

    def __init__(self):
        """Create an empty shell cache."""
        self.shells_: Dict[int, RigidBody] = {}
        self.version_ = None

    def evict_retired(self, geometry: Dict[int, BodyGeometry], version) -> None:
        """Drop cached shells whose uid is gone from the geometry snapshot.

        Description:
            Runs once per geometry epoch per interpreter: the version gate makes
            repeated calls within an epoch a no-op, so the set diff is paid only
            when the body set actually changed. This bounds the cache to live
            uids -- without it a long session that spawns and removes bodies
            leaks a shell per retired uid forever.
        """
        if version == self.version_:
            return

        self.version_ = version
        live = geometry.keys()
        for uid in [uid for uid in self.shells_ if uid not in live]:
            del self.shells_[uid]

    def shells(self, geometry: Dict[int, BodyGeometry], uids: List[int]) -> List[RigidBody]:
        """Return the shells for uids in order, building and caching any missing."""
        result = []
        for uid in uids:
            shell = self.shells_.get(uid)
            if shell is None:
                shell = build_shell(geometry[uid])
                shell.uid = uid
                self.shells_[uid] = shell

            result.append(shell)

        return result
