"""Process-local observational hooks; execute unchanged production Hydra sample task."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dexlearn.task import sample
from dexlearn.network.models.robot_hierarchical import RobotHierarchicalModel
from dexlearn.main import main
original_sample=RobotHierarchicalModel.sample
original_decenter=sample._decenter_robot_pose
original_save=sample.Logger.save_samples
state={}
def sampled(self,data,n):
    forbidden=[k for k in data if k.startswith(('right_hand','left_hand')) or k in ['rand_pose_id','target_type_availability']]
    assert not forbidden,forbidden
    state['scenes']=list(data['scene_path']);state['input_keys']=list(data);state['candidates_requested']=n
    state['centroid']=data['pc_centroid'].detach().cpu().numpy()
    return original_sample(self,data,n)
def decentered(pose,centroid,types):
    before=pose.detach().clone();result=original_decenter(pose,centroid,types)
    c=centroid.reshape(centroid.shape[0],1,1,3)
    assert np.isfinite(result.detach().cpu().numpy()).all()
    import torch
    torch.testing.assert_close(result[...,:3],before[...,:3]+c,rtol=0,atol=0)
    torch.testing.assert_close(result[...,3:29],before[...,3:29],rtol=0,atol=0)
    state['restore_max_error_m']=float((result[...,:3]-(before[...,:3]+c)).abs().max())
    state['selected_shape']=list(result.shape)
    state['centered_pose']=before.detach().cpu().numpy();state['world_pose']=result.detach().cpu().numpy()
    return result
def saved(self,dic,step,paths):
    import torch
    for j,st in enumerate(['pregrasp','grasp','squeeze']):
        np.testing.assert_array_equal(dic[st+'_qpos'].detach().cpu().numpy(),state['world_pose'][:,:,j])
    original_save(self,dic,step,paths)
    dest=Path(self.save_test_dir).parent/'reconstruction_trace';dest.mkdir(exist_ok=True)
    i=len(list(dest.glob('batch_*.json')))
    arrays={k:state[k] for k in ['centroid','centered_pose','world_pose']}
    np.savez_compressed(dest/f'batch_{i:03d}.npz',**arrays)
    (dest/f'batch_{i:03d}.json').write_text(json.dumps({k:v for k,v in state.items() if k not in arrays},indent=2))
RobotHierarchicalModel.sample=sampled
sample._decenter_robot_pose=decentered
sample.Logger.save_samples=saved
if __name__=='__main__':main()
