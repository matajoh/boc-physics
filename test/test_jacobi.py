"""Tests for the Jacobi contact solve core."""

import math

from bocpy import Matrix

from bocphysics import jacobi, xpbd
from bocphysics.bodies import Polygon
from bocphysics.config import DetectionKind
from bocphysics.engine import PhysicsEngine
from bocphysics.physics import Physics
from bocphysics.scene import make_pyramid_scene

GRAVITY = Matrix.vector([0, 9.81])
SUB_DT = (1 / 60) / 20
PHYS = Physics(restitution=0.0, static_friction=0.5, dynamic_friction=0.5)


def build_pile():
    """Three stacked boxes on a static floor; returns (bodies, pairs)."""
    floor = Polygon.create_rectangle(20.0, 2.0, 2.0, (80, 80, 80), is_static=True)
    floor.physics = False
    floor.move_to(Matrix.vector([0.0, 6.0]))
    floor.uid = 0
    boxes = []
    for k in range(3):
        box = Polygon.create_rectangle(2.0, 2.0, 2.0, (50, 120, 200))
        box.physics = True
        box.move_to(Matrix.vector([0.05 * k, 4.0 - 2.0 * k]))
        box.uid = k + 1
        boxes.append(box)
    pairs = [(floor, boxes[0]), (boxes[0], boxes[1]), (boxes[1], boxes[2])]
    return boxes, pairs


def dynamic(engine):
    """Return the engine's dynamic bodies."""
    return [b for b in engine.bodies if b.physics]


def total_ke(bodies):
    """Sum translational and rotational kinetic energy."""
    energy = 0.0
    for b in bodies:
        energy += 0.5 * b.mass * b.linear_velocity.magnitude_squared()
        energy += 0.5 * b.inertia * b.angular_velocity**2
    return energy


def test_accumulate_positions_is_order_independent():
    """Jacobi accumulation over a frozen pose is independent of constraint order."""
    boxes, pairs = build_pile()
    jacobi.integrate_block(boxes, GRAVITY, SUB_DT)
    constraints = xpbd.build_contacts(pairs)
    assert constraints
    prev_pose = {id(b): (b.position.copy(), b.angle) for b in boxes}
    row_of = {id(b): i for i, b in enumerate(boxes)}

    forward = jacobi.new_accumulator(len(boxes))
    jacobi.accumulate_positions(constraints, PHYS, prev_pose, row_of, forward)
    reverse = jacobi.new_accumulator(len(boxes))
    jacobi.accumulate_positions(list(reversed(constraints)), PHYS, prev_pose, row_of, reverse)

    for i in range(len(boxes)):
        assert forward[i, 3] == reverse[i, 3]
        for column in range(3):
            assert abs(forward[i, column] - reverse[i, column]) < 1e-9


def test_accumulator_is_a_block_indexed_by_row():
    """The accumulator is an (N x 4) block and records a positive contribution count."""
    boxes, pairs = build_pile()
    jacobi.integrate_block(boxes, GRAVITY, SUB_DT)
    constraints = xpbd.build_contacts(pairs)
    row_of = {id(b): i for i, b in enumerate(boxes)}
    prev_pose = {id(b): (b.position.copy(), b.angle) for b in boxes}

    acc = jacobi.new_accumulator(len(boxes))
    jacobi.accumulate_positions(constraints, PHYS, prev_pose, row_of, acc)

    assert (acc.rows, acc.columns) == (len(boxes), jacobi.ACC_WIDTH)
    assert sum(acc[i, 3] for i in range(len(boxes))) > 0


def test_jacobi_settles_a_stack():
    """A short box stack settles under the Jacobi solver without exploding."""
    engine = PhysicsEngine(1200, 900, DetectionKind.QUADTREE, show_contacts=False,
                           num_substeps=20)
    engine.physics = PHYS
    for body in make_pyramid_scene(2).build():
        engine.add_body(body)

    for _ in range(300):
        engine.step(1 / 60)

    bodies = dynamic(engine)
    for body in bodies:
        assert math.isfinite(body.position.x)
        assert math.isfinite(body.position.y)
        assert abs(body.position.x) < 50.0
    assert total_ke(bodies) < 1.0
