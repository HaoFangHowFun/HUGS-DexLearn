# Sparse voxel investigation — 2026-10-28

## Root cause

The training dataset and `minkowski_collate_fn` already produce correct sparse geometry.
The first destructive stage was diagnostic `store_sample`: it applied
`value[sample_idx:sample_idx + 1]` to every tensor, including packed sparse tensors.
Their first dimension is voxels (or flattened points), not batch elements.
This truncated `coors`, `feats`, and both mapping arrays to one row.
`merge_samples` also needed to rebase batch IDs and point/voxel mapping offsets.

Affected code: `tests/check_robot_dataloader.py:37` (`store_sample`) and
`:67` (`merge_samples`), duplicated in the local `tests/check_right_full_batch.py`.
The right-full diagnostic now imports the shared corrected helpers.
Production dataset, collation, model, labels, metadata, voxel size, and installed
CUDA/PyTorch/MinkowskiEngine were not changed. Existing local edits were preserved.

## Same-sample evidence before the fix

Seed 0, train_dataset[0], voxel size 0.005 m:

| Stage | Shape | Unique spatial/batched coordinates |
|---|---|---|
| Raw point cloud | (1024, 3) | 919 exact points |
| Dataset coors = pc / 0.005 | (1024, 3) | 919 unquantized coordinates |
| ME sparse_collate | (1024, 4) | 819 |
| ME sparse_quantize | (819, 4) | 819 |
| DexLearn minkowski_collate_fn | (819, 4) | 819 |
| Four copies, normal collation | (3276, 4) | 3276 |
| Old diagnostic store/merge, first sample | (1, 4) | 1 |

The raw NumPy arrays were float64, CPU, contiguous, stride (3, 1).
Point-cloud bounds were approximately [-0.08519242, -0.09000818, -0.09250577]
to [0.08433690, 0.09089266, 0.09036152] m. Raw coors ranged from
[-17.03848, -18.00164, -18.50115] to [16.86738, 18.17853, 18.07230].
`coors == point_clouds / voxel_size` was verified within the same sample.
Collation/quantization returned contiguous CPU int32 coordinates, stride (4, 1),
with range [0, -18, -19, -19] to [0, 16, 18, 18]. Inputs were not mutated.
Correct index/inverse shapes were (819,) / (1024,). The old diagnostic changed
both to (1,) and retained exactly coordinate [0, 11, 13, -5].

Installed ME sparse_collate floors floating coordinates into int32;
sparse_quantize deduplicates those integer coordinates. Its observed behavior
matches the PyTorch reference. There is no demonstrated double-quantization bug.

## Verification commands

Run from HUGS-DexLearn, with the existing virtual environment:

```bash
unset PYTHONPATH LD_LIBRARY_PATH
export HUGS_DATASET_ROOT=/home/hao-fang/hugs-dataset
.venv/bin/python -m unittest discover -s tests -p test_robot_sparse_batch.py -v

.venv/bin/python tests/check_right_full_batch.py \
  data=shadowMulti exp_name=right_full_smoke_test \
  "data.paths.grasp_path=$HUGS_DATASET_ROOT/BimanBODex_single_hand_demo_v2/shadow" \
  data.point_cloud.source=complete data.point_cloud.preload=false \
  data.robot.metadata_group=right_full data.runtime.num_workers=0 +check_max_iter=20

OMP_NUM_THREADS=4 .venv/bin/python tests/check_robot_sparse_training.py \
  data=shadowMulti algo=robotMultiHierar \
  "data.paths.grasp_path=$HUGS_DATASET_ROOT/BimanBODex_single_hand_demo_v2/shadow" \
  data.point_cloud.source=complete data.point_cloud.preload=false \
  data.robot.metadata_group=right_full
```

All 3 regression tests passed. They exercise actual RobotMultiDexDataset.__getitem__
with only disk/grasp loading stubbed, then real MinkowskiEngine collation.
Coverage includes B=1/B=4, duplicate points, nonzero sample selection, reordered
samples across batches, both map reconstruction identities, feature alignment,
source immutability during selection/merge, and target-type collection.
The real dataset and GPU checks below cover actual disk data without stubs.

## Real right_full diagnostic after fix

```text
=== RIGHT_FULL BATCH PASSED ===

=== POINT CLOUD DIAGNOSTICS ===
Min: [-0.08519242 -0.09000818 -0.09250577]
Max: [0.08433689 0.09089266 0.09036152]
Extent (meters): [0.16952932 0.18090084 0.18286729]
Unique sparse voxels: 819
Voxel size: 0.005
target_type_availability: (1, 5)
scene_type_counts: (1, 5)
right_hand_trans: (1, 1, 3, 3)
right_hand_rot: (1, 1, 3, 3, 3)
right_hand_joint: (1, 1, 3, 22)
left_hand_trans: (1, 1, 3, 3)
left_hand_rot: (1, 1, 3, 3, 3)
left_hand_joint: (1, 1, 3, 22)
left_hand_fixed: (1,)
path: list
rand_pose_id: (1,)
pc_centroid: (1, 1, 3)
point_clouds: (1, 1024, 3)
grasp_type_id: (1,)
coors: (819, 4)
quantize2original: (1024,)
feats: (819, 3)
original2quantize: (819,)

=== SAME BATCH COMPARISON ===
PyTorch unique voxels: 819
Minkowski unique voxels: 819
DexLearn batch voxels: 819
Coordinates range: [  0 -18 -19 -19] [ 0 16 18 18]
```

## Real robot GPU training smoke test

```text
Batch shapes: {'target_type_availability': (2, 5), 'scene_type_counts': (2, 5), 'right_hand_trans': (2, 1, 3, 3), 'right_hand_rot': (2, 1, 3, 3, 3), 'right_hand_joint': (2, 1, 3, 22), 'left_hand_trans': (2, 1, 3, 3), 'left_hand_rot': (2, 1, 3, 3, 3), 'left_hand_joint': (2, 1, 3, 22), 'left_hand_fixed': (2,), 'rand_pose_id': (2,), 'pc_centroid': (2, 1, 3), 'point_clouds': (2, 1024, 3), 'grasp_type_id': (2,), 'coors': (1330, 4), 'feats': (1330, 3), 'original2quantize': (1330,), 'quantize2original': (2048,)}
Losses: {'loss_type': 0.7312303781509399, 'loss_diffusion': 0.46005699038505554, 'metric_trans_right': 0.004462090786546469, 'metric_rot_right': 0.16671767830848694, 'metric_joint_right': 0.08224213123321533, 'abs_dis_joint_right': 0.33851444721221924, 'metric_trans_left': 0.041607148945331573, 'metric_rot_left': 0.16673749685287476, 'metric_joint_left': 3.6005108086101245e-06, 'abs_dis_joint_left': 0.0020947072189301252, 'loss_trans': 0.02303461916744709, 'loss_rot': 0.16672758758068085, 'loss_joint': 0.04112286493182182, 'abs_dis_joint': 0.1703045815229416}
FORWARD/BACKWARD PASSED: 133 finite gradient tensors; joint_num=44
```

This was one training-mode forward/backward pass with B=2 and no optimizer step.
The batch had 1,330 sparse voxels and a 2,048-element inverse map. The inspected
right-only batch had left_hand_fixed=True and zero left-hand joint targets;
the model retained joint_num=44. All 133 populated gradient tensors were finite.
This verifies one-batch compatibility, not long-run convergence.

## Actual split counts and remaining scope

- Train: 5 grouped scene/grasp files, 80 individual grasps (19 + 17 + 9 + 16 + 19).
- Eval (valid_split/test.json): 1 grouped file, 11 individual grasps.
- Total: 91 individual grasps across 6 grouped files.

No remaining blocker was found for sparse geometry or the small model smoke test.
The original bimanual viewer still expects left-hand metadata and multiple grasp
types. Those pre-existing viewer assumptions are separate from this sparse fix;
no metadata was fabricated and no large-scale training was launched.

## Targeted helper diff

This diff is against the pre-edit local snapshot, excluding earlier user edits.

```diff
--- before/tests/check_robot_dataloader.py
+++ after/tests/check_robot_dataloader.py
@@ -31,21 +31,45 @@
     return default if value is None else value


+SPARSE_KEYS = {"coors", "feats", "original2quantize", "quantize2original"}
+
+
 def store_sample(batch_data, sample_idx):
+    """Select a dense sample and rebase its packed sparse coordinates/maps."""
     sample = {}
     for key, value in batch_data.items():
+        if key in SPARSE_KEYS:
+            continue
         if torch.is_tensor(value):
             sample[key] = value[sample_idx : sample_idx + 1].clone()
         elif isinstance(value, list):
             sample[key] = [value[sample_idx]]
         else:
             sample[key] = value
+
+    if "coors" in batch_data:
+        # Sparse rows are voxels, not batch entries. Preserve their ordering.
+        mask = batch_data["coors"][:, 0] == sample_idx
+        sample["coors"] = batch_data["coors"][mask].clone()
+        sample["coors"][:, 0] = 0
+        sample["feats"] = batch_data["feats"][mask].clone()
+        point_num = batch_data["point_clouds"].shape[1]
+        point_start = sample_idx * point_num
+        sample["original2quantize"] = batch_data["original2quantize"][mask] - point_start
+        inverse = batch_data["quantize2original"]
+        voxel_ids = torch.nonzero(mask, as_tuple=True)[0]
+        local_ids = torch.full((len(mask),), -1, dtype=inverse.dtype, device=inverse.device)
+        local_ids[voxel_ids] = torch.arange(len(voxel_ids), device=inverse.device, dtype=inverse.dtype)
+        sample["quantize2original"] = local_ids[inverse[point_start : point_start + point_num]]
     return sample


 def merge_samples(samples):
+    """Merge selected samples, offsetting voxel and original-point indices."""
     merged = {}
     for key in samples[0]:
+        if key in SPARSE_KEYS:
+            continue
         first_value = samples[0][key]
         if torch.is_tensor(first_value):
             merged[key] = torch.cat([sample[key] for sample in samples], dim=0)
@@ -53,6 +77,20 @@
             merged[key] = [sample[key][0] for sample in samples]
         else:
             merged[key] = [sample[key] for sample in samples]
+
+    if "coors" in samples[0]:
+        sparse = {key: [] for key in SPARSE_KEYS}
+        point_offset = voxel_offset = 0
+        for batch_idx, sample in enumerate(samples):
+            coors = sample["coors"].clone()
+            coors[:, 0] = batch_idx
+            sparse["coors"].append(coors)
+            sparse["feats"].append(sample["feats"])
+            sparse["original2quantize"].append(sample["original2quantize"] + point_offset)
+            sparse["quantize2original"].append(sample["quantize2original"] + voxel_offset)
+            point_offset += sample["point_clouds"].shape[1]
+            voxel_offset += len(coors)
+        merged.update({key: torch.cat(values, dim=0) for key, values in sparse.items()})
     return merged


```
