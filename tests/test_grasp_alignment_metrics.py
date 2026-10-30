"""Analytical checks for physical distance units and rigid-transform validation."""
import unittest
import numpy as np
import trimesh
from omegaconf import OmegaConf
from visualize_grasp_alignment import distance_stats, training_validation, RobotMultiDexDataset


class AlignmentMetricsTest(unittest.TestCase):
    def test_known_surface_distance_in_meters_and_common_rigid_transform(self):
        mesh = trimesh.creation.box(extents=[.2, .2, .2])
        points = np.array([[.11, 0, 0], [0, .12, 0], [0, 0, .13]])
        metrics = distance_stats(mesh, points)
        self.assertAlmostEqual(metrics['mean'], .02, places=12)
        self.assertAlmostEqual(metrics['max'], .03, places=12)
        transform = trimesh.transformations.rotation_matrix(.7, [0, 0, 1])
        transform[:3, 3] = [2, -3, .4]
        mesh.apply_transform(transform)
        moved = points @ transform[:3, :3].T + transform[:3, 3]
        self.assertAlmostEqual(distance_stats(mesh, moved)['mean'], .02, places=12)
        # Moving only the query cloud must change the result (no implicit fitting).
        self.assertGreater(distance_stats(mesh, moved + [1, 0, 0])['mean'], .5)

    def test_training_transform_preserves_active_palm_frame_and_left_sentinel(self):
        ds = RobotMultiDexDataset.__new__(RobotMultiDexDataset)
        ds.mode = 'train'
        ds.config = OmegaConf.create({'num_points': 1024})
        grouped = {}
        for stage in ('pregrasp', 'grasp', 'squeeze'):
            grouped[f'{stage}_global_pose'] = np.array([[.1, -.2, .3, 1, 0, 0, 0]])
            grouped[f'{stage}_joint_pos'] = np.zeros((1, 22))
        pc = np.random.default_rng(1).uniform(-1, 1, (4096, 3)).astype(np.float32)
        result = training_validation(ds, grouped, 0, pc, np.ones(3), np.eye(3), np.zeros(3), 37)
        self.assertTrue(result['fixed_left_unchanged'])
        self.assertTrue(result['right_joints_unchanged'])
        np.testing.assert_allclose(result['palm_frame_point_invariance_max_abs_error_m_by_stage'], 0, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
