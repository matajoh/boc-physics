"""Tests for declarative static scenes."""

from bocphysics.scene import DEFAULT_SCENE, OPEN_BOX, Scene, StaticBody


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


def test_static_body_build_regular_polygon():
    spec = StaticBody("regular_polygon", (10, 20, 30), (1, 2),
                      num_sides=6, radius=2.0)
    body = spec.build()
    assert body.inv_mass == 0
    assert len(body.vertices) == 6
    assert body.position.x == 1 and body.position.y == 2


def test_static_body_build_circle():
    spec = StaticBody("circle", (10, 20, 30), (3, 4), radius=1.5)
    body = spec.build()
    assert body.inv_mass == 0
    assert body.inv_inertia == 0
    assert body.radius == 1.5
    assert body.position.x == 3 and body.position.y == 4


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
