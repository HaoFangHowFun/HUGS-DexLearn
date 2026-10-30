"""Run one real robot batch through sparse encoding and forward/backward."""
import sys
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexlearn.dataset import InfLoader, create_dataset, minkowski_collate_fn
from dexlearn.network.models import RobotHierarchicalModel
from dexlearn.utils.util import set_seed


@hydra.main(config_path='../dexlearn/config', config_name='base', version_base=None)
def main(config):
    set_seed(config.seed)
    dataset = create_dataset(config, mode='train')
    batch = InfLoader(
        DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=minkowski_collate_fn),
        config.device,
    ).get()
    print('Batch shapes:', {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)})
    assert batch['coors'].shape[0] > batch['point_clouds'].shape[0]
    assert batch['feats'].shape[0] == batch['coors'].shape[0]
    assert batch['quantize2original'].numel() == batch['point_clouds'].shape[0] * 1024
    assert config.algo.model.head.joint_num == 44
    right_only = batch['left_hand_fixed'].bool()
    assert torch.count_nonzero(batch['left_hand_joint'][right_only]) == 0

    model = RobotHierarchicalModel(config.algo.model).to(config.device).train()
    results = model(batch)
    loss = sum(config.algo.loss_weight[k] * v for k, v in results.items() if k in config.algo.loss_weight)
    print('Losses:', {k: float(v.detach()) for k, v in results.items()})
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)
    print(f'FORWARD/BACKWARD PASSED: {len(grads)} finite gradient tensors; joint_num=44')


if __name__ == '__main__':
    main()
