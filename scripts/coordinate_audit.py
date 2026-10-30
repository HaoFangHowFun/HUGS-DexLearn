"""Read-only coordinate audit and isolated Viser viewer; no model inference/rollout.
Run with ../HUGS-DexGraspBench/.venv/bin/python from HUGS-DexLearn.
"""
import argparse, ast, copy, hashlib, json, os, socket, sys, threading, time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
from transforms3d.quaternions import quat2mat

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO.parent / 'HUGS-DexGraspBench'
DATA = Path(os.environ.get('HUGS_DATASET_ROOT', Path.home() / 'hugs-dataset'))
OUT = REPO / 'output/coordinate_audit_2026-10-29'
TRAIN = DATA / 'BimanBODex_shadow_100k_v1/shadow/right_full'
LEARNED = BENCH / 'output/learned_right_full_100k_v1_right_full_shadow'
BODEX = BENCH / 'output/bodex_matched_test_v1_full_right_full_shadow'
SAMPLES = REPO / 'output/shadowMulti_robotMultiHierar_right_full_100k_v1/tests/step_050000/shadowMulti'
STAGES = ('pregrasp', 'grasp', 'squeeze')
sys.path[:0] = [str(REPO), str(BENCH / 'src'), str(REPO / 'tests')]
from dexlearn.dataset.robot_multidex import RobotMultiDexDataset
from dexlearn.task import visualize as vis
from dexlearn.utils.rot import numpy_quaternion_to_matrix
from util.hand_util import MjHO
from task.convert_format import _convert_learning_stage_single, _load_learning_metadata
from visualize_grasp_alignment import geometry_from_mujoco, distance_stats

HASHES = {}
def read(path):
    path = Path(path).resolve()
    HASHES.setdefault(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    return np.load(path, allow_pickle=True).item()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)))

def functions(path, names, namespace):
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def select_cases():
    train_ids=set(json.loads((DATA/'object/DGN_2k/valid_split/train.json').read_text()))
    objects=sorted(p for p in TRAIN.iterdir() if p.is_dir() and p.name in train_ids)
    rng=np.random.default_rng(123); cases=[]
    for oi in sorted(rng.choice(len(objects),6,replace=False)):
        files=sorted(objects[oi].rglob('*.npy')); f=files[len(files)//2]; d=read(f)
        index=int(np.flatnonzero(d['succ_flag'])[0])
        cases.append(dict(label='TRAIN success / '+objects[oi].name,scene=str(DATA/d['scene_path']),
                          variants=[dict(label='Training BODex success',path=str(f),index=index,kind='train')]))
    good={}; failures={}
    for f in sorted((BODEX/'evaluation').rglob('*.npy')):
        d=read(f)
        if bool(d['succ_flag']): good.setdefault(d['scene_path'],f)
    for f in sorted((LEARNED/'evaluation').rglob('*.npy')):
        d=read(f)
        if d['delta_pos']==100 and d['scene_path'] in good:
            obj=d['scene_path'].split('/')[3]; failures.setdefault(obj,f)
    for obj,f in sorted(failures.items()):
        d=read(f); src=SAMPLES/f.relative_to(LEARNED/'evaluation')
        cases.append(dict(label='MATCHED control vs prediction / '+obj,scene=str(DATA/d['scene_path']),variants=[
            dict(label='BODex evaluated success',path=str(good[d['scene_path']]),index=None,kind='eval'),
            dict(label='DexLearn raw prediction',path=str(src),index=None,kind='prediction'),
            dict(label='DexLearn evaluated failure',path=str(f),index=None,kind='eval')]))
    assert len(cases)==16
    dump(OUT/'cases.json',cases)
    return cases


def poses(variant):
    d=read(variant['path']); index=variant['index']
    if index is not None:
        names=list(d['joint_names'][index]); q=np.stack([d[s+'_qpos'][index] for s in STAGES])
    else:
        names=list(d.get('joint_names',json.loads((TRAIN/'metadata.json').read_text())['joint_names']))
        q=np.stack([d[s+'_qpos'][:29] for s in STAGES])
    return d,names,q.astype(np.float64)


def scene_data(case):
    scene_path=Path(case['scene']); s=read(scene_path); obj=s['scene'][s['task']['obj_name']]
    objroot=DATA/'object/DGN_2k/processed_data'/s['task']['obj_name']
    pcpath=objroot/'complete_point_cloud.npy'; raw=np.load(pcpath)
    HASHES.setdefault(str(pcpath),hashlib.sha256(pcpath.read_bytes()).hexdigest())
    # Independent float64 reference, not the loader's rotation helper.
    R=quat2mat(obj['pose'][3:7]); scale=np.asarray(obj['scale']); t=np.asarray(obj['pose'][:3])
    pc=(raw.astype(float)*scale)@R.T+t
    meshpath=(scene_path.parent/obj['file_path']).resolve()
    HASHES.setdefault(str(meshpath),hashlib.sha256(meshpath.read_bytes()).hexdigest())
    mesh=trimesh.load(meshpath,force='mesh',process=False); mesh.vertices=(mesh.vertices*scale)@R.T+t
    ho=MjHO(obj_path=str(objroot),obj_scale=float(scale[0]),obj_density=700,
            mj_arena_memory_bytes=67108864,hand_xml_path=str(BENCH/'assets/hand/shadow/right_hand_v2.xml'),
            hand_mocap=True,exclude_table_contact=None,friction_coef=[.8,.02],has_floor_z0=True)
    return s,obj,raw,pc,mesh,ho


def set_pose(ho,names,q,obj):
    order=[names.index(n) for n in ho.get_joint_names()[1:]]
    mq=q.copy(); mq[7:]=q[7:][order]; ho.reset_pose_qpos(mq,obj['pose'],set_ctrl=False)
    if not hasattr(ho, '_qpos2ctrl_matrix'): ho._init_after_first_fk()
    return mq


def contact_depths(ho,mq,obj):
    hc,sc=ho.get_contact_info(mq,obj['pose'])
    return {'object_penetration_m':max([0.]+[-float(c['contact_dist']) for c in hc]),
            'self_penetration_m':max([0.]+[-float(c['contact_dist']) for c in sc])}


def dataset_checks(case,s,obj,raw,pc):
    ds=RobotMultiDexDataset.__new__(RobotMultiDexDataset); ds.mode='train'
    ds.config=OmegaConf.create(dict(name="shadowMulti",num_points=1024,pc_centering=True,rotation_aug=False,pc_noise_aug=False,object_path=str(DATA/'object/DGN_2k')))
    _,scale,R,t=ds._extract_object_meta(s,case['scene'])
    indices=np.random.default_rng(123).choice(len(raw),1024,replace=True)
    local=raw[indices].astype(np.float32); world=ds._transform_complete_pc(local,scale,R,t)
    vp=vis.transform_complete_pc(local,*vis.extract_object_meta(s,case['scene'])[1:])
    result={'loader_vs_independent_max_m':float(abs(world-pc[indices]).max()),'loader_vs_independent_mean_m':float(abs(world-pc[indices]).mean()),
            'viewer_vs_loader_max_m':float(abs(vp-world).max()),'object_scale':obj['scale'],'object_pose':obj['pose']}
    if case['variants'][0]['kind']=='train':
        v=case['variants'][0]; d=read(v['path']); ds.obj_id_lst=['audit']; ds.grasp_path_dict={'audit':[v['path']]}
        ds.grasp_scene_id_dict={v['path']:s['scene_id']}; ds.pc_path_dict={s['scene_id']:[str(DATA/'object/DGN_2k/processed_data'/s['task']['obj_name']/'complete_point_cloud.npy')]}
        ds.pc_source='complete'; ds.preload_point_clouds=False; ds.type_availability_enabled=False; ds.scene_cfg_dict={s['scene_id']:s}
        ret={}
        with patch('numpy.random.randint',return_value=v['index']),patch('numpy.random.choice',return_value=indices):
            _,actual,_=ds._load_train_data(ret)
        result['actual_train_loader_pc_max_m']=float(abs(actual-world).max())
        q=np.stack([d[st+'_global_pose'][v['index']] for st in STAGES])
        np.testing.assert_allclose(ret['right_hand_trans'][0],q[:,:3],atol=1e-7)
    else:
        d,names,_=poses(case['variants'][0]); q=np.stack([d[st+'_global_pose'] for st in STAGES])
        ret={'right_hand_trans':q[None,:,:3].copy(),'right_hand_rot':numpy_quaternion_to_matrix(q[None,:,3:7]),
             'right_hand_joint':np.stack([d[st+'_joint_pos'] for st in STAGES])[None], 'left_hand_fixed':True,
             'left_hand_trans':np.tile([0,0,-.5],(1,3,1)),'left_hand_rot':np.tile(np.eye(3),(1,3,1,1))}
    original=copy.deepcopy(ret); centered=ds._apply_pc_centering(world.copy(),ret,None)
    centroid=ret['pc_centroid']; result['pc_center_inverse_max_m']=float(abs(centered+centroid-world).max())
    result['wrist_center_inverse_max_m']=float(abs(ret['right_hand_trans']+centroid-original['right_hand_trans']).max())
    theta=.647; rz=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    with patch('numpy.random.uniform',return_value=theta): aug=ds._apply_rotation_aug(centered.copy(),ret)
    errors=[]
    for i in range(3):
        before=(world-original['right_hand_trans'][0,i])@original['right_hand_rot'][0,i]
        after=(aug-ret['right_hand_trans'][0,i])@ret['right_hand_rot'][0,i]
        errors.append(float(abs(before-after).max()))
    result['augmentation_palm_frame_invariance_max_m']=max(errors)
    ds.mode='test'; ds.test_cfg_lst=[case['scene']]; ds.grasp_type_lst=['0_any']; ds.pc_path_dict={}; ds.pc_source='complete'; ds.preload_point_clouds=False; ds.complete_pc_folder=str(DATA/'object/DGN_2k/processed_data')
    testret={}
    with patch('numpy.random.choice',return_value=indices): _,testpc=ds._load_test_data(0,testret)
    result['actual_test_loader_pc_max_m']=float(abs(testpc-world).max())
    ds._apply_pc_centering(testpc,testret,None)
    ns=functions(REPO/'dexlearn/task/sample.py',{'_expand_centroid','_decenter_robot_pose'},{'torch':torch})
    restored=[]
    metadata=_load_learning_metadata(case['scene'],'shadow','right_full')
    for i in range(3):
        # A known good world wrist is centered, exported, then formatted.
        right=np.r_[q[i,:3]-testret['pc_centroid'][0],q[i,3:7],original['right_hand_joint'][0,i]]
        two=np.r_[right,np.zeros(29)].astype(np.float32)
        out=ns['_decenter_robot_pose'](torch.tensor(two)[None,None,None],torch.tensor(testret['pc_centroid'])[None],torch.tensor([[3]]))
        out=_convert_learning_stage_single(out.numpy()[0,0,0],metadata)
        restored.append(float(abs(out[:3]-q[i,:3]).max()))
    result['known_pose_decenter_format_inverse_max_m']=max(restored)
    return result


def audit_case(case):
    s,obj,raw,pc,mesh,ho=scene_data(case)
    result={'label':case['label'],'scene':case['scene'],'transform':dataset_checks(case,s,obj,raw,pc),
            'pc_bounds_m':np.array([pc.min(0),pc.max(0)]),'mesh_bounds_m':mesh.bounds,
            'pc_to_mesh_distance':distance_stats(mesh,pc[np.linspace(0,len(pc)-1,256,dtype=int)]),'variants':[]}
    for v in case['variants']:
        d,names,qs=poses(v); stages={}
        for i,stage in enumerate(STAGES):
            mq=set_pose(ho,names,qs[i],obj); palm=ho.model.body('child-rh_palm').id
            expected=qs[i,:7]
            fkpos=ho.data.xpos[palm].copy(); fkR=ho.data.xmat[palm].reshape(3,3).copy()
            ref=d.get(stage+'_global_pose'); ref=ref[v['index']] if ref is not None and v['index'] is not None else ref
            jointids=[ho.model.joint("child-"+n).id for n in ho.get_joint_names()[1:]]
            bounds=ho.model.jnt_range[jointids]; violation=np.maximum(bounds[:,0]-mq[7:],mq[7:]-bounds[:,1]).clip(0)
            stages[stage]={'quaternion_norm':float(np.linalg.norm(expected[3:])),
              'fk_vs_input_translation_max_m':float(abs(fkpos-expected[:3]).max()),
              'fk_vs_input_rotation_max':float(abs(fkR-quat2mat(expected[3:])).max()),
              'fk_vs_saved_global_translation_max_m':None if ref is None else float(abs(fkpos-ref[:3]).max()),
              'fk_vs_saved_global_rotation_max':None if ref is None else float(abs(fkR-quat2mat(ref[3:])).max()),
              'joint_out_of_range_count':int((violation>1e-5).sum()),'max_joint_violation_rad':float(violation.max()),
              'wrist_to_object_centroid_m':float(np.linalg.norm(fkpos-pc.mean(0))),**contact_depths(ho,mq,obj)}
        result['variants'].append(dict(source=v,stages=stages))
    return result


def serve(cases,port):
    import viser
    # Fail if occupied instead of silently moving to another port.
    with socket.socket() as sock: sock.bind(('127.0.0.1',port))
    server=viser.ViserServer(host='127.0.0.1',port=port)
    server.scene.set_up_direction('+z'); server.scene.add_frame('/world',axes_length=.08,axes_radius=.001)
    chosen=server.gui.add_dropdown('Dataset / case',options=[c['label'] for c in cases])
    stage=server.gui.add_dropdown('Stage',options=STAGES,initial_value='grasp')
    variant=server.gui.add_dropdown('Pose source',options=[v['label'] for v in cases[0]['variants']])
    mesh_on=server.gui.add_checkbox('Object mesh',initial_value=True)
    pc_on=server.gui.add_checkbox('Complete point cloud',initial_value=True)
    info=server.gui.add_markdown('World coordinates; meters. Full right hand via MuJoCo FK. No rollout.')
    cache={}; handles=[]; lock=threading.RLock()
    def refresh(reset=False):
        with lock:
            idx=list(chosen.options).index(chosen.value); c=cases[idx]
            options=[v['label'] for v in c['variants']]
            if list(variant.options)!=options:
                variant.options=options; variant.value=options[0]
            if idx not in cache: cache[idx]=scene_data(c)
            s,obj,raw,pc,mesh,ho=cache[idx]
            v=c['variants'][options.index(variant.value)]; d,names,qs=poses(v)
            set_pose(ho,names,qs[STAGES.index(stage.value)],obj); hand=geometry_from_mujoco(ho,True)
            for h in handles: h.remove()
            handles.clear()
            handles.append(server.scene.add_mesh_simple('/hand',vertices=hand.vertices,faces=hand.faces,color=(230,120,145)))
            handles.append(server.scene.add_mesh_simple('/object',vertices=mesh.vertices,faces=mesh.faces,color=(150,165,180),opacity=.35,visible=mesh_on.value))
            handles.append(server.scene.add_point_cloud('/pointcloud',points=pc[::max(1,len(pc)//4096)],colors=(255,165,0),point_size=.0015,visible=pc_on.value))
            info.content=f'**{v["label"]}**\n\nScene: `{s["scene_id"]}`\n\nSource: `{v["path"]}`\n\nIndex: {v["index"]}; stage: {stage.value}. World frame, meters. Static FK; no simulation or inference. Orange: cloud; gray: object; pink: complete right hand.'
            center=pc.mean(0)
            if reset:
                for client in server.get_clients().values(): client.camera.look_at=tuple(center); client.camera.position=tuple(center+np.array([.4,-.45,.35]))
    @chosen.on_update
    def _(event): refresh(True)
    for control in (stage,variant,mesh_on,pc_on): control.on_update(lambda event:refresh())
    @server.on_client_connect
    def _(client): refresh(True)
    refresh()
    print(f'VIEWER_READY http://127.0.0.1:{port}',flush=True)
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt: server.stop()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--serve',action='store_true'); p.add_argument('--port',type=int,default=8087); p.add_argument('--audit',action='store_true'); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.audit:
        cases=select_cases(); results=[]
        for i,c in enumerate(cases):
            results.append(audit_case(c)); print('AUDITED',i+1,c['label'],flush=True)
        dump(OUT/'case_metrics.json',results)
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in HASHES.items())
        dump(OUT/'read_artifact_sha256.json',HASHES)
        print('READ_ARTIFACTS_UNCHANGED',len(HASHES),flush=True)
    else:
        cases=json.loads((OUT/'cases.json').read_text())
    if args.serve: serve(cases,args.port)

if __name__=='__main__': main()
