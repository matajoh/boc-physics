"""Tests for the physics engine step driver."""

import random

from bocpy import Matrix

from bocphysics.bodies import Circle, Polygon
from bocphysics.config import DetectionKind, PhysicsMode
from bocphysics.engine import PhysicsEngine


def make_engine() -> PhysicsEngine:
    """Create a windowless engine with friction physics and quadtree detection."""
    return PhysicsEngine(1200, 900, PhysicsMode.FRICTION,
                         DetectionKind.QUADTREE, show_contacts=False)


def test_overlapping_static_bodies_do_not_crash():
    """Two overlapping static bodies must not trigger collision response."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    wall = Polygon.create_rectangle(2, 24, 2.0, (100, 100, 100), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    engine.add_body(wall.move_to(Matrix.vector([-14, -2])))

    floor_before = (floor.position.x, floor.position.y)
    wall_before = (wall.position.x, wall.position.y)

    engine.step(1 / 60)

    # static bodies are never integrated, so they stay exactly put
    assert (floor.position.x, floor.position.y) == floor_before
    assert (wall.position.x, wall.position.y) == wall_before


def populate_random(engine, count: int, seed: int):
    """Drop a deterministic spread of dynamic circles into the engine."""
    rng = random.Random(seed)
    for _ in range(count):
        x = rng.uniform(-11, 11)
        y = rng.uniform(-13, 5)
        engine.add_body(Circle.create(rng.uniform(0.6, 1.2), 2.0, (200, 100, 50))
                        .move_to(Matrix.vector([x, y])))


def test_islands_partition_dynamic_bodies_disjointly():
    """Every dynamic body belongs to exactly one island."""
    for seed in range(20):
        engine = make_engine()
        floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
        engine.add_body(floor.move_to(Matrix.vector([0, 10])))
        populate_random(engine, 25, seed)
        engine.update_swept_aabbs(1 / 60)
        engine.collisions.clear()
        engine.broad_phase()
        islands = engine.build_islands()

        seen = []
        dynamic = [b for b in engine.bodies if b.physics]
        for island in islands:
            for body in island.bodies:
                seen.append(body)

        # the islands together cover every dynamic body exactly once
        assert len(seen) == len(dynamic)
        assert len(set(map(id, seen))) == len(dynamic)
        assert set(map(id, seen)) == set(map(id, dynamic))


def test_island_pairs_have_a_dynamic_member():
    """No island holds a static-static candidate pair."""
    engine = make_engine()
    floor = Polygon.create_rectangle(30, 2, 2.0, (0, 100, 0), is_static=True)
    wall = Polygon.create_rectangle(2, 24, 2.0, (100, 100, 100), is_static=True)
    engine.add_body(floor.move_to(Matrix.vector([0, 10])))
    engine.add_body(wall.move_to(Matrix.vector([-14, -2])))
    populate_random(engine, 20, seed=7)
    engine.update_swept_aabbs(1 / 60)
    engine.collisions.clear()
    engine.broad_phase()
    islands = engine.build_islands()

    for island in islands:
        for a, b in island.pairs:
            assert a.physics or b.physics


def test_isolated_body_forms_a_singleton_island():
    """A lone falling body becomes its own one-body island."""
    engine = make_engine()
    engine.add_body(Circle.create(1.0, 2.0, (200, 100, 50)).move_to(Matrix.vector([0, 0])))
    engine.update_swept_aabbs(1 / 60)
    engine.collisions.clear()
    engine.broad_phase()
    islands = engine.build_islands()

    assert len(islands) == 1
    assert len(islands[0].bodies) == 1
    assert islands[0].pairs == []


def test_isolated_body_falls_under_gravity():
    """A singleton island still integrates so the body accelerates downward."""
    engine = make_engine()
    body = Circle.create(1.0, 2.0, (200, 100, 50))
    engine.add_body(body.move_to(Matrix.vector([0, 0])))

    engine.step(1 / 60)

    # gravity is +y in the engine's y-down world, so the body moves downward
    assert body.position.y > 0
    assert body.linear_velocity.y > 0
