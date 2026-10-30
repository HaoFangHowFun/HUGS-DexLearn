"""Bounded array/FK diagnostics; no learned inference or MuJoCo stepping."""
import ast,json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
import sys
# Reuse the existing pure-Python einops package; benchmark dependencies stay first.
sys.path.append(str(Path(__file__).resolve().parents[1]/".venv/lib/python3.10/site-packages"))
import numpy as np
import torch
from einops import rearrange
from mr_utils.pytorch3d import rotation_conversions as pttf
from coordinate_audit import *
from dexlearn.utils.rot import proper_svd


def main():
    cases=json.loads((OUT/'cases.json').read_text()); s,obj,raw,pc,mesh,ho=scene_data(cases[0])
    metadata=json.loads((TRAIN/'metadata.json').read_text()); names=metadata['joint_names']
    bounds=np.array([ho.model.jnt_range[ho.model.joint('child-'+n).id] for n in names])
    summary={}; scene_centers={}; offsets={}
    for label,root in [('DexLearn_raw',SAMPLES),('DexLearn_evaluated',LEARNED/'evaluation'),('BODex_formatted',BODEX/'graspdata'),('BODex_evaluated',BODEX/'evaluation')]:
        values=defaultdict(list); scene_wrist=defaultdict(list)
        for f in sorted(root.rglob('*.npy')):
            d=read(f); order=[list(d.get('joint_names',names)).index(n) for n in names]
            scene=str(DATA/d['scene_path']); scene_wrist[scene].append(d['grasp_qpos'][:3])
            for st in STAGES:
                q=np.asarray(d[st+'_qpos'][:29]); joint=q[7:][order]
                v=np.maximum(bounds[:,0]-joint,joint-bounds[:,1]).clip(0)
                values[st+'_outside'].append(int((v>1e-5).any())); values[st+'_max_violation'].append(float(v.max()))
                values[st+'_quat_norm_error'].append(float(abs(np.linalg.norm(q[3:7])-1)))
                values[st+'_pre_delta'].append(float(np.linalg.norm(q[:3]-d['grasp_qpos'][:3])))
        summary[label]={k:({'count':int(sum(v)),'fraction':float(np.mean(v))} if k.endswith('_outside') else {'median':float(np.median(v)),'p95':float(np.quantile(v,.95)),'max':float(max(v))}) for k,v in values.items()}
        if label in ('DexLearn_raw','BODex_formatted'): offsets[label]={k:np.mean(v,axis=0) for k,v in scene_wrist.items()}
    # Population translation association, not candidate correspondence or a correction.
    scenes=sorted(offsets['DexLearn_raw']); centroids=[]; deltas=[]
    for path in scenes:
        cfg=read(path); obj=cfg['scene'][cfg['task']['obj_name']]
        pcfile=DATA/'object/DGN_2k/processed_data'/cfg['task']['obj_name']/'complete_point_cloud.npy'
        local=np.load(pcfile).astype(float)
        c=(local.mean(0)*obj['scale'])@quat2mat(obj['pose'][3:7]).T+obj['pose'][:3]
        centroids.append(c); deltas.append(offsets['DexLearn_raw'][path]-offsets['BODex_formatted'][path])
    c=np.array(centroids); delta=np.array(deltas)
    summary['scene_mean_wrist_difference']={'mean_m':delta.mean(0),'rms_m':float(np.sqrt((delta**2).mean())),
        'rms_if_add_full_cloud_centroid_m':float(np.sqrt(((delta+c)**2).mean())),
        'rms_if_subtract_full_cloud_centroid_m':float(np.sqrt(((delta-c)**2).mean())),
        'caveat':'BODex and DexLearn candidates are not pose-corresponded. Full-cloud centroid is not the exact original random 1024-point inference centroid.'}
    # Exercise the actual decoder with oracle MLP values and the true last-stage pose.
    tree=ast.parse((REPO/'dexlearn/network/final_layers/diffusion.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DiffusionBiRT_MLPRTJ')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_sample_single_hand')
    ns={'torch':torch,'rearrange':rearrange,'proper_svd':proper_svd,'pttf':pttf}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_decoder','exec'),ns)
    d,n,q=poses(cases[0]['variants'][0]); trans=torch.tensor(q[:,:3],dtype=torch.float32); rot=pttf.quaternion_to_matrix(torch.tensor(q[:,3:7],dtype=torch.float32)); joints=torch.tensor(q[:,7:],dtype=torch.float32)
    centroid=torch.tensor(pc.mean(0),dtype=torch.float32); trans=trans-centroid
    oracle=torch.cat([trans[:2].reshape(-1),rot[:2].reshape(-1),joints.reshape(-1)])[None]
    last=torch.cat([rot[-1].reshape(-1),trans[-1]])[None]
    decoded=ns['_sample_single_hand'](NS(cfg=NS(traj_length=3)),torch.zeros((1,1)),last,lambda x:oracle,1)[0,0]
    summary['oracle_stage_decoder']={'translation_max_m':float(abs(decoded[:,:3]-trans).max()),'joint_max_rad':float(abs(decoded[:,7:]-joints).max()),'rotation_matrix_max':float(abs(pttf.quaternion_to_matrix(decoded[:,3:7])-rot).max())}
    # Actual Logger writes only an isolated new audit artifact.
    from dexlearn.utils.logger import Logger
    dec=functions(REPO/'dexlearn/task/sample.py',{'_expand_centroid','_decenter_robot_pose'},{'torch':torch})['_decenter_robot_pose']
    two=torch.cat([decoded,torch.zeros_like(decoded)],-1)[None,None]
    world=dec(two,centroid[None,None],torch.tensor([[3]]))
    logger=Logger.__new__(Logger); logger.save_test_dir=str(OUT/'export_control'); logger.dataset_root=str(DATA)
    dic={st+'_qpos':world[:,:,i] for i,st in enumerate(STAGES)}; dic.update(scene_path=[str(DATA/d['scene_path'])],grasp_type_id=torch.tensor([3]))
    logger.save_samples(dic,0,['oracle.npy'])
    saved=read(OUT/'export_control/step_000000/oracle_0.npy')
    summary['logger_roundtrip']={st:float(abs(saved[st+'_qpos'][:29]-world[0,0,i,:29].numpy()).max()) for i,st in enumerate(STAGES)}
    dump(OUT/'population_metrics.json',summary)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in HASHES.items())
    dump(OUT/'population_read_sha256.json',HASHES)
    print(json.dumps(summary,indent=2,default=lambda x:x.tolist()))

if __name__=='__main__': main()
