"""Tests for declarative static scenes."""

import random

from bocphysics.scene import (BodySpec, DEFAULT_SCENE, Generator, GeneratorSpec,
                              make_default_drop_scene, make_golden_scene,
                              make_open_box_drop_scene, make_pachinko_scene,
                              make_pyramid_scene, make_stack_scene, OPEN_BOX,
                              Scene, ShapeCategory)


def _unit_emitter(rate, seed=0, limit=0):
    return GeneratorSpec((ShapeCategory("circle"),), x_range=(0.0, 0.0),
                         y_range=(0.0, 0.0), size_range=(1.0, 1.0), rate=rate,
                         limit=limit, seed=seed)


def test_default_scene_matches_legacy_hard_wired_bodies():
    bodies = DEFAULT_SCENE.build()
    assert len(bodies) == 3
    assert all(body.inv_mass == 0 for body in bodies)
    floor = bodies[0]
    assert floor.position.x == 0 and floor.position.y == 10
    assert floor.angle == 0
    assert bodies[1].angle > 0
    assert bodies[2].angle < 0


def test_open_box_has_floor_and_two_walls():
    bodies = OPEN_BOX.build()
    assert len(bodies) == 3
    assert all(body.inv_mass == 0 for body in bodies)
    xs = sorted(body.position.x for body in bodies)
    assert xs[0] < 0 < xs[-1]


def test_body_spec_build_regular_polygon():
    spec = BodySpec("regular_polygon", (10, 20, 30), (1, 2),
                    num_sides=6, radius=2.0)
    body = spec.build()
    assert body.inv_mass == 0
    assert len(body.vertices) == 6
    assert body.position.x == 1 and body.position.y == 2


def test_body_spec_build_circle():
    spec = BodySpec("circle", (10, 20, 30), (3, 4), radius=1.5)
    body = spec.build()
    assert body.inv_mass == 0
    assert body.inv_inertia == 0
    assert body.radius == 1.5
    assert body.position.x == 3 and body.position.y == 4


def test_body_spec_build_dynamic_has_mass():
    spec = BodySpec("rectangle", (10, 20, 30), (0, 0), width=2, height=2)
    body = spec.build(is_static=False)
    assert body.inv_mass > 0
    assert body.inv_inertia > 0


def test_scene_round_trips_through_dict():
    for scene in (DEFAULT_SCENE, OPEN_BOX):
        restored = Scene.from_dict(scene.to_dict())
        assert restored == scene


def test_load_scene_by_builtin_name():
    assert Scene.load("default") == DEFAULT_SCENE
    assert Scene.load("open_box") == OPEN_BOX


def test_save_and_load_scene_file_round_trips(tmp_path):
    path = str(tmp_path / "scene.json")
    OPEN_BOX.save(path)
    assert Scene.load(path) == OPEN_BOX


def test_scene_with_dynamics_round_trips_through_dict():
    scene = make_pyramid_scene(3)
    assert Scene.from_dict(scene.to_dict()) == scene


def test_stack_scene_has_floor_and_dynamic_column():
    scene = make_stack_scene(5)
    assert len(scene.statics) == 1
    assert len(scene.dynamics) == 5
    bodies = scene.build()
    assert bodies[0].inv_mass == 0
    assert all(body.inv_mass > 0 for body in bodies[1:])
    assert all(body.position.x == 0 for body in bodies[1:])


def test_pyramid_scene_rows_taper():
    scene = make_pyramid_scene(4)
    assert len(scene.dynamics) == 4 + 3 + 2 + 1


def test_golden_scene_is_deterministic():
    first = make_golden_scene()
    second = make_golden_scene()
    assert first == second
    assert len(first.dynamics) == 24


def test_builtin_scenario_scenes_are_loadable():
    for name in ("stack", "pyramid", "golden"):
        scene = Scene.load(name)
        assert scene.name == name
        assert scene.dynamics


def test_generator_spec_round_trips_through_dict():
    spec = GeneratorSpec((ShapeCategory("circle"), ShapeCategory("regular_polygon", 2.0, 5)),
                         x_range=(-3.0, 3.0), y_range=(-1.0, 1.0), size_range=(0.8, 1.6),
                         rate=5.0, color=(10, 20, 30), randomize_color=True,
                         density=3.0, limit=12, seed=7)
    assert GeneratorSpec.from_dict(spec.to_dict()) == spec


def test_pachinko_scene_round_trips_through_dict():
    scene = make_pachinko_scene(4)
    assert Scene.from_dict(scene.to_dict()) == scene


def test_pachinko_scene_has_walls_and_one_generator():
    scene = make_pachinko_scene(5)
    assert scene.name == "pachinko"
    assert len(scene.statics) > 3
    assert scene.dynamics == ()
    assert len(scene.generators) == 1
    assert scene.generators[0].shapes[0].kind == "circle"
    assert scene.view_height > 30
    assert all(body.inv_mass == 0 for body in scene.build())


def test_scene_view_height_round_trips_through_dict():
    scene = make_pachinko_scene(3)
    restored = Scene.from_dict(scene.to_dict())
    assert restored == scene
    assert restored.view_height == scene.view_height


def test_pachinko_is_builtin_and_loadable():
    scene = Scene.load("pachinko")
    assert scene.name == "pachinko"
    assert scene.generators


def test_generator_emits_near_poisson_rate():
    rng = random.Random(7)
    for _ in range(20):
        rate = rng.uniform(5, 40)
        dt = 1 / rng.choice([30, 60, 120])
        steps = int(rng.uniform(40, 120) / dt)
        gen = Generator(_unit_emitter(rate, seed=rng.randint(0, 9999)))
        total = sum(len(gen.update(dt)) for _ in range(steps))
        expected = rate * steps * dt
        assert abs(total - expected) <= 5 * expected ** 0.5 + 5


def test_generator_limit_caps_total_emitted():
    gen = Generator(_unit_emitter(100.0, limit=7))
    total = sum(len(gen.update(1 / 60)) for _ in range(1000))
    assert total == 7


def test_generator_seeded_streams_are_identical():
    spec = GeneratorSpec((ShapeCategory("circle"), ShapeCategory("rectangle"),
                          ShapeCategory("regular_polygon", num_sides=5)),
                         x_range=(-3.0, 3.0), y_range=(-1.0, 1.0), size_range=(1.0, 2.0),
                         rate=20.0, randomize_color=True, seed=99)
    a, b = Generator(spec), Generator(spec)
    pa = [(body.position.x, body.position.y, body.color, body.inv_mass)
          for _ in range(120) for body in a.update(1 / 60)]
    pb = [(body.position.x, body.position.y, body.color, body.inv_mass)
          for _ in range(120) for body in b.update(1 / 60)]
    assert pa and pa == pb


def test_circle_size_is_a_diameter():
    gen = Generator(GeneratorSpec((ShapeCategory("circle"),), x_range=(0.0, 0.0),
                                  y_range=(0.0, 0.0), size_range=(2.0, 2.0), rate=1000.0))
    bodies = []
    while not bodies:
        bodies = gen.update(1 / 60)
    assert bodies[0].radius == 1.0


def test_rectangle_size_draws_width_and_height_independently():
    gen = Generator(GeneratorSpec((ShapeCategory("rectangle"),), x_range=(0.0, 0.0),
                                  y_range=(0.0, 0.0), size_range=(1.0, 3.0),
                                  rate=1000.0, seed=3))
    nonsquare = False
    for _ in range(300):
        for body in gen.update(1 / 60):
            xs = [v.x for v in body.vertices]
            ys = [v.y for v in body.vertices]
            if abs((max(xs) - min(xs)) - (max(ys) - min(ys))) > 1e-9:
                nonsquare = True
    assert nonsquare


def test_drop_scenes_reuse_statics_and_add_emitters():
    for make, base in ((make_default_drop_scene, DEFAULT_SCENE),
                       (make_open_box_drop_scene, OPEN_BOX)):
        scene = make()
        assert scene.statics == base.statics
        assert scene.dynamics == ()
        assert len(scene.generators) == 1
        kinds = {shape.kind for shape in scene.generators[0].shapes}
        assert kinds == {"circle", "rectangle", "regular_polygon"}
        assert all(body.inv_mass == 0 for body in scene.build())


def test_drop_scenes_round_trip_and_are_loadable():
    for name in ("default_drop", "open_box_drop"):
        scene = Scene.load(name)
        assert scene.name == name
        assert scene.generators
        assert Scene.from_dict(scene.to_dict()) == scene
