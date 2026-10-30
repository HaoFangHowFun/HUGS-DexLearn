"""Joint-order and single-hand regression coverage for visualization IK."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dexlearn.task.visualize import solve_stage_pose_batch


@pytest.mark.parametrize("both", [False, True])
def test_ik_uses_named_joints_and_only_available_wrists(both):
    chain_names = ["la_x", "rh_b", "ra_x", "lh_b", "rh_a", "lh_a"]
    metadata_names = ["rh_a", "rh_b"] + (["lh_a", "lh_b"] if both else [])
    calls = []

    def solve(wrist, matrix):
        calls.append((wrist, matrix.clone()))
        q = torch.full((2, 6), -999., dtype=torch.float64)
        q[:, chain_names.index("ra_x" if wrist == "rh_palm" else "la_x")] = 0.25
        return {"q": q, "success": torch.tensor([True, False])}

    assets = {
        "robot_helper": SimpleNamespace(robot_joint_names=chain_names, solve_ik_batch=solve),
        "joint_names": metadata_names,
        # Reverse wrist order to ensure side selection does not depend on position.
        "wrist_link_names": ["lh_palm", "rh_palm"] if both else ["rh_palm"],
        "right_hand_indices": [0, 1], "left_hand_indices": [2, 3] if both else [],
    }
    stage = np.tile([1, 2, 3, 1, 0, 0, 0, 11, 12,
                     4, 5, 6, 1, 0, 0, 0, 21, 22], (2, 1))
    pose, right_ok, left_ok = solve_stage_pose_batch(stage, assets, "cpu")
    expected = [0.25 if both else 0, 12, 0.25, 22 if both else 0, 11, 21 if both else 0]
    np.testing.assert_allclose(pose[:, 7:], np.tile(expected, (2, 1)))
    np.testing.assert_array_equal(right_ok, [True, False])
    np.testing.assert_array_equal(left_ok, [both, False])
    assert [wrist for wrist, _ in calls] == (["rh_palm", "lh_palm"] if both else ["rh_palm"])
    np.testing.assert_allclose(calls[0][1][:, :3, 3], np.tile([1, 2, 3], (2, 1)))
    if both:
        np.testing.assert_allclose(calls[1][1][:, :3, 3], np.tile([4, 5, 6], (2, 1)))
    np.testing.assert_array_equal(pose[:, :7], np.tile([0, 0, 0, 1, 0, 0, 0], (2, 1)))
