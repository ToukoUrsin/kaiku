"""Fast geometry checks for the offline baseline."""
import numpy as np

from kaiku.policy.baseline import SourceDemo, segments, symmetry_delta, transform_xy
from kaiku.sim.scene import Scene


def _source(n=3):
    scene = Scene()
    T = 180
    parts = np.zeros((T, n, 7), dtype=np.float32)
    names = ["obj_blue_triangle_0", "obj_red_triangle_0", "obj_yellow_cube_0"][:n]
    layout = []
    for j, name in enumerate(names):
        p = np.array([0.27 + .02 * j, -.12 + .08 * j, scene.rest_z])
        parts[:, j, :3] = p
        parts[:, j, 3] = 1
        layout.append({"name": name, "pos": [*p], "yaw": 0.0})
        parts[50 + j * 35 :, j, 2] += .03
        mapping = {"red": "left", "yellow": "center", "blue": "right"}
        parts[90 + j * 35 :, j, :2] = scene.bin_centers[mapping[scene.by_name[name].color]][:2]
        parts[90 + j * 35 :, j, 2] = scene.rest_z
    row = {"episode": 0, "layout": layout,
           "mapping": {"red": "left", "yellow": "center", "blue": "right"}}
    return SourceDemo(None, row, np.zeros((T, 28)), np.zeros((T, 28)), parts, (0, T))


def test_shape_symmetry_uses_the_shortest_equivalent_grasp():
    assert np.isclose(symmetry_delta("cube", 0.0, np.pi / 2 + .1), .1)
    assert np.isclose(symmetry_delta("triangle", 0.0, 2 * np.pi / 3 + .1), .1)
    assert np.isclose(symmetry_delta("cylinder", 0.0, 3 * np.pi), -np.pi)


def test_object_transform_rotates_about_the_object():
    got = transform_xy(np.array([[1.0, 0.0]]), np.zeros(2), np.array([2.0, 3.0]), np.pi / 2)
    assert np.allclose(got, [[2.0, 4.0]])


def test_segments_have_reach_and_carry_for_each_part():
    got = segments(_source())
    assert len(got) == 6
    assert [x.kind for x in got] == ["reach_grasp", "carry_drop"] * 3
    assert all(x.start < x.end for x in got)
    assert all(x.target_frame == ("object" if x.kind == "reach_grasp" else "bin") for x in got)
