"""Tests for declarative static scenes."""

from bocphysics.scene import (BodySpec, DEFAULT_SCENE, make_golden_scene,
                              make_pyramid_scene, make_stack_scene, OPEN_BOX,
                              Scene)


def test_default_scene_matches_legacy_hard_wired_bodies():
    bodies = DEFAULT_SCENE.build()
    assert len(bodies) == 3
    # all scene bodies are static
    assert all(body.inv_mass == 0 for body in bodies)
    # the floor sits where the old hard-wired body did
    floor = bodies[0]
    assert floor.position.x == 0 and floor.position.y == 10
    assert floor.angle == 0
    # the two ledges carry the old angles
    assert bodies[1].angle > 0
    assert bodies[2].angle < 0


def test_open_box_has_floor_and_two_walls():
    bodies = OPEN_BOX.build()
    assert len(bodies) == 3
    assert all(body.inv_mass == 0 for body in bodies)
    xs = sorted(body.position.x for body in bodies)
    # one wall on each side, floor in the middle
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
    # the floor is static, every box is dynamic
    assert bodies[0].inv_mass == 0
    assert all(body.inv_mass > 0 for body in bodies[1:])
    # the column is vertically aligned
    assert all(body.position.x == 0 for body in bodies[1:])


def test_pyramid_scene_rows_taper():
    scene = make_pyramid_scene(4)
    # a 4-level pyramid stacks 4 + 3 + 2 + 1 boxes
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
