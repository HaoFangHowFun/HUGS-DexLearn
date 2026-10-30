"""Exercise the cached loader without starting robot IK or a viewer."""

import ast
import inspect

import numpy as np
import pytest
from omegaconf import OmegaConf

from dexlearn.task import visualize


@pytest.fixture
def pc_loader(tmp_path):
    # Compile the actual nested helper with its session state supplied explicitly.
    tree = ast.parse(inspect.getsource(visualize.task_visualize_robot))
    helper = next(node for node in tree.body[0].body if isinstance(node, ast.FunctionDef)
                  and node.name == "load_cached_robot_visualization_pc")
    config = OmegaConf.create({"test_data": {
        "object_path": str(tmp_path), "pc_path": "vision_data", "num_points": 8,
        "pc_source": "complete",
    }})
    namespace = dict(vars(visualize), config=config, robot_pc_cache={})
    exec(compile(ast.Module(body=[helper], type_ignores=[]), inspect.getfile(visualize), "exec"), namespace)
    return namespace[helper.name], config


def scene(scene_id, translation):
    pose = np.eye(4)
    pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    pose[:3, 3] = translation
    return {"scene_id": scene_id, "object": {
        "name": "object_a", "scale": [2, 3, 4], "pose": pose,
    }}


def test_complete_world_transform_and_scene_cache(pc_loader, tmp_path):
    loader, config = pc_loader
    path = tmp_path / "processed_data/object_a/complete_point_cloud.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.tile([1., 2., 3.], (10, 1)))
    first_scene = scene("object_a/pose0", [10, 20, 30])
    # No sample payload or pc_path is required for complete clouds.
    first = loader("complete_point_cloud_0.npy", first_scene)
    np.testing.assert_allclose(first, np.tile([4, 22, 42], (8, 1)))
    assert loader("complete_point_cloud_1.npy", first_scene) is first
    second = loader("complete_point_cloud_0.npy", scene("object_a/pose1", [20, 30, 40]))
    np.testing.assert_allclose(second, first + 10)
    assert second is not first


@pytest.mark.parametrize("explicit_source", [True, False])
def test_partial_preserves_world_coordinates(pc_loader, tmp_path, explicit_source):
    loader, config = pc_loader
    if explicit_source:
        config.test_data.pc_source = "partial"
    else:
        del config.test_data.pc_source
    path = tmp_path / "vision_data/object_a/pose0/partial_pc_0.npy"
    path.parent.mkdir(parents=True)
    np.save(path, np.tile([1., 2., 3.], (10, 1)))
    cfg = {"scene_id": "object_a/pose0"}
    result = loader("partial_pc_0_0.npy", cfg)
    np.testing.assert_array_equal(result, np.tile([1, 2, 3], (8, 1)))
    assert loader("partial_pc_0_1.npy", cfg) is result
