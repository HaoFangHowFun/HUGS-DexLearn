"""Regression for robot dataset -> collation -> diagnostic batch selection."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from dexlearn.dataset import minkowski_collate_fn
from dexlearn.dataset.robot_multidex import RobotMultiDexDataset
from check_robot_dataloader import store_sample, merge_samples, collect_target_grasp_batch


class RobotSparseBatchTest(unittest.TestCase):
    def make_sample(self, seed):
        # Exercise the actual __getitem__, mocking only disk/grasp loading.
        dataset = RobotMultiDexDataset.__new__(RobotMultiDexDataset)
        dataset.mode = 'train'
        dataset.config = OmegaConf.create({'pc_centering': False})
        dataset.sc_voxel_size = 0.005
        rng = np.random.default_rng(seed)
        pc = rng.uniform(-0.09, 0.09, (1024, 3))
        pc[1::4] = pc[::4]  # Include duplicate coordinates to test both maps.
        with patch.object(dataset, '_load_train_data', return_value=('3_right_full', pc, None)):
            return dataset[0]

    def assert_sparse_valid(self, batch):
        points = batch['point_clouds']
        b, n, _ = points.shape
        reference = torch.cat([
            torch.cat([torch.full((n, 1), i, dtype=torch.int32), torch.floor(pc / 0.005).int()], dim=1)
            for i, pc in enumerate(points)
        ])
        coors = batch['coors']
        unique, inverse = batch['original2quantize'], batch['quantize2original']
        self.assertEqual(tuple(coors.shape), (len(torch.unique(reference, dim=0)), 4))
        self.assertGreater(len(coors), 500 * b)
        self.assertEqual(tuple(batch['feats'].shape), (len(coors), 3))
        self.assertEqual(tuple(unique.shape), (len(coors),))
        self.assertEqual(tuple(inverse.shape), (b * n,))
        torch.testing.assert_close(coors[inverse], reference)
        torch.testing.assert_close(reference[unique], coors)
        torch.testing.assert_close(inverse[unique], torch.arange(len(coors)))
        torch.testing.assert_close(batch['feats'], points.reshape(-1, 3)[unique].to(batch['feats'].dtype))

    def test_dataset_collation_single_and_multiple_samples(self):
        for size in (1, 4):
            with self.subTest(size=size):
                samples = [self.make_sample(i) for i in range(size)]
                saved = copy.deepcopy(samples)
                batch = minkowski_collate_fn(copy.deepcopy(samples))
                self.assert_sparse_valid(batch)
                for sample, original in zip(samples, saved):
                    np.testing.assert_array_equal(sample['coors'], original['coors'])

    def test_select_nonzero_sample_and_merge_reordered_batches(self):
        batches = [minkowski_collate_fn([self.make_sample(i + offset) for i in range(4)]) for offset in (0, 10)]
        saved = copy.deepcopy(batches)
        selected = [store_sample(batches[0], 3), store_sample(batches[1], 1), store_sample(batches[0], 0)]
        for sample in selected:
            self.assert_sparse_valid(sample)
        self.assert_sparse_valid(merge_samples(selected))
        for batch, original in zip(batches, saved):
            for key in batch:
                torch.testing.assert_close(batch[key], original[key])

    def test_target_collection_preserves_sparse_geometry(self):
        batch = minkowski_collate_fn([self.make_sample(i) for i in range(4)])
        batch['grasp_type_id'] = torch.tensor([1, 2, 3, 4])
        class Loader:
            def get(self):
                return batch
        selected = collect_target_grasp_batch(Loader(), [3, 1], max_iter=1)
        self.assertEqual(selected['grasp_type_id'].tolist(), [3, 1])
        self.assert_sparse_valid(selected)


if __name__ == '__main__':
    unittest.main()
