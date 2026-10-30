"""Actual __getitem__/sparse-collation audit; run using DexLearn .venv Python.
Only writes output/dataloader_coordinate_audit; never performs model inference.
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO.parent / 'HUGS-DexGraspBench/.venv/lib/python3.10/site-packages'))
from coordinate_audit import *
import random
from contextlib import ExitStack
from transforms3d.quaternions import mat2quat
from dexlearn.dataset import minkowski_collate_fn, InfLoader
from torch.utils.data import DataLoader

DEBUG_OUT = REPO / 'output/dataloader_coordinate_audit'


def configuration():
    root = OmegaConf.create({'data_root':str(DATA), 'asset_root':str(REPO/'assets'),
                             'data':OmegaConf.load(REPO/'dexlearn/config/data/shadowMulti.yaml')})
    cfg = OmegaConf.create(OmegaConf.to_container(root.data, resolve=True))
    cfg.paths.grasp_path = str(TRAIN.parent)
    cfg.point_cloud.source = 'complete'; cfg.point_cloud.preload = False
    cfg.robot.metadata_group = 'right_full'
    return cfg


def inverse(pc, trans, rot, centroid, augmentation, noise):
    """Row points/translations; column rotation matrices. Noise is recorded, not guessed."""
    return ((pc-noise) @ augmentation + centroid,
            trans @ augmentation + centroid,
            augmentation.T @ rot)


def geodesic(a, b):
    # Project rounded float32 matrices onto SO(3) before measuring angle.
    def project(r):
        u, _, vt = np.linalg.svd(r)
        fix = np.ones(3); fix[-1] = np.linalg.det(u @ vt)
        return (u * fix) @ vt
    delta = project(a) @ project(b).T
    return float(np.arctan2(np.linalg.norm([delta[2,1]-delta[1,2], delta[0,2]-delta[2,0], delta[1,0]-delta[0,1]])/2,
                           (np.trace(delta)-1)/2))


def qposes(trans, rot, joints):
    return np.stack([np.r_[trans[i],mat2quat(rot[i]),joints[i]] for i in range(3)])


def capture(ds, case, mode, seed):
    """Real DataLoader calls real __getitem__; hooks only observe or fix random selection."""
    v=case['variants'][0]; source=read(v['path']); scene=read(case['scene'])
    obj=scene['task']['obj_name']; scene_id=scene['scene_id']
    assert obj in ds.obj_id_lst and v['path'] in ds.grasp_path_dict[obj]
    pc_path=ds.pc_path_dict[scene_id][0]
    raw=np.load(pc_path); HASHES.setdefault(pc_path,hashlib.sha256(Path(pc_path).read_bytes()).hexdigest())
    ds.config.rotation_aug=mode!='no_aug'; ds.config.pc_noise_aug=mode=='rotation_noise'
    trace={}; choice=np.random.choice; uniform=np.random.uniform
    center=ds._apply_pc_centering; noise_fn=ds._apply_pc_noise_aug
    def sample(*args, **kwargs):
        result=choice(*args,**kwargs); trace['indices']=result.copy(); return result
    def angle(*args, **kwargs):
        result=uniform(*args,**kwargs); trace['angle']=float(result); return result
    def centering(pc, ret, grasp):
        trace['precenter']=pc.copy(); return center(pc,ret,grasp)
    def noise(pc):
        result=noise_fn(pc); trace['noise']=result.astype(float)-pc.astype(float); return result
    py_state=random.getstate(); np_state=np.random.get_state()
    random.seed(seed); np.random.seed(seed)
    try:
        with ExitStack() as stack:
            stack.enter_context(patch('random.choice',side_effect=[obj,v['path'],pc_path]))
            stack.enter_context(patch('numpy.random.randint',return_value=v['index']))
            stack.enter_context(patch('numpy.random.choice',side_effect=sample))
            stack.enter_context(patch('numpy.random.uniform',side_effect=angle))
            stack.enter_context(patch.object(ds,'_apply_pc_centering',side_effect=centering))
            stack.enter_context(patch.object(ds,'_apply_pc_noise_aug',side_effect=noise))
            item=next(iter(DataLoader(ds,batch_size=1,num_workers=0,collate_fn=lambda items:items)))[0]
    finally:
        random.setstate(py_state); np.random.set_state(np_state)
    assert item['path']==v['path'] and item['rand_pose_id']==v['index']
    assert bool(source['succ_flag'][v['index']])
    theta=trace.get('angle',0.); c,s=np.cos(theta),np.sin(theta)
    A=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    eps=trace.get('noise',np.zeros_like(item['point_clouds']))
    op=scene['scene'][obj]; reference=(raw[trace['indices']].astype(float)*np.asarray(op['scale']))@quat2mat(op['pose'][3:]).T+op['pose'][:3]
    original=np.stack([source[st+'_global_pose'][v['index']] for st in STAGES])
    joints=np.stack([source[st+'_joint_pos'][v['index']] for st in STAGES])
    original_R=np.stack([quat2mat(q[3:]) for q in original])
    recovered=inverse(item['point_clouds'],item['right_hand_trans'][0],item['right_hand_rot'][0],item['pc_centroid'],A,eps)
    arrays={'pc':item['point_clouds'],'trans':item['right_hand_trans'][0],'rot':item['right_hand_rot'][0],
            'joints':item['right_hand_joint'][0],'centroid':item['pc_centroid'],'augmentation':A,'noise':eps,
            'indices':trace['indices'],'reference_pc':reference,'precenter':trace['precenter'],
            'original_trans':original[:,:3],'original_rot':original_R,'original_joints':joints,
            'recovered_pc':recovered[0],'recovered_trans':recovered[1],'recovered_rot':recovered[2]}
    metrics={'precenter_pc_max_m':float(abs(trace['precenter']-reference).max()),
             'inverse_pc_max_m':float(abs(recovered[0]-reference).max()),
             'centered_mean_norm_m':float(np.linalg.norm(item['point_clouds'].mean(0))),
             'noise_max_m':float(abs(eps).max()),'angle_rad':theta,'stages':{}}
    _,objdata,_,_,_,ho=scene_data(case)
    names=list(source['joint_names'][v['index']]); metadata=json.loads((TRAIN/'metadata.json').read_text())
    assert names==metadata['joint_names'] and len(names)==22
    for i,st in enumerate(STAGES):
        q=qposes(*recovered[1:],arrays['joints'])[i]
        set_pose(ho,names,q,objdata)
        bodies=['rh_palm','rh_ffdistal','rh_mfdistal','rh_rfdistal','rh_lfdistal','rh_thdistal']
        actual=np.stack([ho.data.xpos[ho.model.body('child-'+n).id].copy() for n in bodies])
        set_pose(ho,names,np.r_[original[i],joints[i]],objdata)
        expected=np.stack([ho.data.xpos[ho.model.body('child-'+n).id].copy() for n in bodies])
        metrics['stages'][st]={'translation_max_m':float(abs(recovered[1][i]-original[i,:3]).max()),
           'rotation_geodesic_rad':geodesic(recovered[2][i],original_R[i]),
           'joint_max_rad':float(abs(arrays['joints'][i]-joints[i]).max()),
           'palm_fingertip_fk_max_m':float(abs(actual-expected).max())}
    assert np.array_equal(item['left_hand_trans'],np.tile([0,0,-.5],(1,3,1)))
    assert np.array_equal(item['left_hand_rot'],np.tile(np.eye(3),(1,3,1,1)))
    assert not item['left_hand_joint'].any()
    return item,arrays,metrics


def audit():
    DEBUG_OUT.mkdir(parents=True,exist_ok=True)
    cases=json.loads((OUT/'cases.json').read_text())[:6]
    ds=RobotMultiDexDataset(configuration(),'train',sc_voxel_size=.005)
    OmegaConf.save(ds.config,DEBUG_OUT/'diagnostic_config.yaml')
    inventory={'all_files':0,'train_files':0,'all_records':0,'train_records':0,'all_success':0}
    for f in sorted(TRAIN.glob('*/*/*.npy')):
        d=read(f); n=len(d['succ_flag'])
        inventory['all_files']+=1; inventory['all_records']+=n
        inventory['all_success']+=int(np.asarray(d['succ_flag']).sum())
        if f.parts[-3] in ds.obj_id_lst:
            inventory['train_files']+=1; inventory['train_records']+=n
    dump(DEBUG_OUT/'inventory.json',inventory)
    for p in [TRAIN/'metadata.json',DATA/'object/DGN_2k/valid_split/train.json',REPO/'dexlearn/dataset/robot_multidex.py',REPO/'dexlearn/dataset/__init__.py',REPO/'dexlearn/config/data/shadowMulti.yaml']:
        HASHES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[]; items=[]
    for mode in ('no_aug','rotation','rotation_noise'):
        for i,case in enumerate(cases):
            item,arrays,metrics=capture(ds,case,mode,123+i)
            filename=f'case_{i}_{mode}.npz'; np.savez_compressed(DEBUG_OUT/filename,**arrays)
            rows.append({'case_index':i,'case':case,'mode':mode,'seed':123+i,'artifact':filename,
                         'keys':{k:{'shape':list(np.shape(v)),'dtype':str(getattr(v,'dtype',type(v).__name__))} for k,v in item.items()},'metrics':metrics})
            items.append(item); print('CAPTURED',i,mode,metrics,flush=True)
    # Exercise the production sparse collator AND InfLoader's float32 cast.
    before=copy.deepcopy(items)
    collated=next(iter(DataLoader(copy.deepcopy(items),batch_size=len(items),collate_fn=minkowski_collate_fn,num_workers=0)))
    batch=InfLoader(DataLoader(copy.deepcopy(items),batch_size=len(items),collate_fn=minkowski_collate_fn,num_workers=0),'cpu').get()
    for key in ('point_clouds','right_hand_trans','right_hand_rot','right_hand_joint','pc_centroid'):
        np.testing.assert_array_equal(collated[key].numpy(),np.stack([x[key] for x in before]))
    sparse=collated['coors'].numpy()[collated['quantize2original'].numpy()]
    expected=np.concatenate([np.column_stack([np.full(len(x['coors']),i),np.floor(x['coors'])]) for i,x in enumerate(before)])
    np.testing.assert_array_equal(sparse,expected)
    np.testing.assert_array_equal(collated['feats'].numpy(),np.concatenate([x['feats'] for x in before])[collated['original2quantize'].numpy()].astype(collated['feats'].numpy().dtype))
    cast_errors=[]
    for i,row in enumerate(rows):
        with np.load(DEBUG_OUT/row['artifact']) as z: arrays={k:z[k] for k in z.files}
        pc,t,R=inverse(batch['point_clouds'][i].numpy(),batch['right_hand_trans'][i,0].numpy(),batch['right_hand_rot'][i,0].numpy(),batch['pc_centroid'][i].numpy(),arrays['augmentation'],arrays['noise'])
        cast_errors.append({'pc_max_m':float(abs(pc-arrays['reference_pc']).max()),'wrist_max_m':float(abs(t-arrays['original_trans']).max()),
                            'rotation_max_rad':max(geodesic(R[j],arrays['original_rot'][j]) for j in range(3))})
    dump(DEBUG_OUT/'items.json',rows)
    dump(DEBUG_OUT/'batch.json',{'batch_size':len(items),'keys':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in batch.items() if isinstance(v,torch.Tensor)},
        'collation_values_unchanged':True,'sparse_mapping_pass':True,'post_InfLoader_inverse':cast_errors})
    # One true test __getitem__; oracle substitution only, never invoke model.sample.
    cfg=OmegaConf.create(dict(name='shadowMulti',object_path=str(DATA/'object/DGN_2k'),grasp_path=str(TRAIN.parent),split_path='valid_split',pc_path='vision_data/three_realsense_d435_random4096',pc_source='complete',preload_point_clouds=False,num_points=1024,pc_centering=True,test_split='test',test_scene_cfg='tabletop_ur10e/**.npy',mini_test=False,grasp_type_lst=['3_right_full'],pc_runtime_scale=1.))
    testds=RobotMultiDexDataset(cfg,'test',sc_voxel_size=.005)
    case=json.loads((OUT/'cases.json').read_text())[6]; idx=testds.test_cfg_lst.index(case['scene'])
    np.random.seed(4321); testitem=testds[idx]
    d,names,q=poses(case['variants'][1]); cent=testitem['pc_centroid']
    two=np.stack([d[st+'_qpos'] for st in STAGES]).astype(np.float32)
    centered=two.copy(); centered[:,:3]-=cent
    dec=functions(REPO/'dexlearn/task/sample.py',{'_expand_centroid','_decenter_robot_pose'},{'torch':torch})['_decenter_robot_pose']
    restored=dec(torch.tensor(centered)[None,None],torch.tensor(cent)[None],torch.tensor([[3]])).numpy()[0,0]
    from dexlearn.utils.logger import Logger
    logger=Logger.__new__(Logger); logger.save_test_dir=str(DEBUG_OUT/'export_control'); logger.dataset_root=str(DATA)
    dic={st+'_qpos':torch.tensor(restored[i])[None,None] for i,st in enumerate(STAGES)}
    dic.update(scene_path=[case['scene']],grasp_type_id=torch.tensor([3]))
    logger.save_samples(dic,0,['heldout_oracle.npy'])
    saved=read(DEBUG_OUT/'export_control/step_000000/heldout_oracle_0.npy')
    dump(DEBUG_OUT/'heldout.json',{'scene':case['scene'],'raw_prediction':case['variants'][1]['path'],'dataset_index':idx,
        'centroid':cent,'test_centered_mean_norm_m':float(np.linalg.norm(testitem['point_clouds'].mean(0))),
        'oracle_export_max_m':float(abs(restored[:,:3]-two[:,:3]).max()),
        'logger_max_error':max(float(abs(saved[st+'_qpos']-restored[i]).max()) for i,st in enumerate(STAGES)),
        'limitation':'Controlled new test centroid; historical model output and historical centroid cannot be replayed from saved npy. No inference run.'})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in HASHES.items())
    dump(DEBUG_OUT/'read_sha256.json',HASHES)
    print('AUDIT_COMPLETE',flush=True)


def serve(port,initial_case,initial_mode):
    import viser
    rows=json.loads((DEBUG_OUT/'items.json').read_text()); cache={}
    with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    server=viser.ViserServer(host='127.0.0.1',port=port); server.scene.set_up_direction('+z')
    chosen=server.gui.add_dropdown('Exact training item',options=[f'{i}: '+rows[i]['case']['label'] for i in range(6)],initial_value=f'{initial_case}: '+rows[initial_case]['case']['label'])
    mode=server.gui.add_dropdown('Preprocessing',options=['no_aug','rotation','rotation_noise'],initial_value=initial_mode)
    frame=server.gui.add_dropdown('Frame',options=['Model input','Recovered world'])
    stage=server.gui.add_dropdown('Stage',options=STAGES,initial_value='grasp')
    overlay=server.gui.add_checkbox('Original ground truth overlay',initial_value=True)
    info=server.gui.add_markdown(''); handles=[]; lock=threading.RLock()
    def refresh(reset=False):
        with lock:
            index=int(chosen.value.split(':')[0]); row=next(r for r in rows if r['case_index']==index and r['mode']==mode.value)
            with np.load(DEBUG_OUT/row['artifact']) as z:a={k:z[k] for k in z.files}
            if index not in cache:cache[index]=scene_data(row['case'])
            _,obj,_,_,mesh,ho=cache[index]; names=json.loads((TRAIN/'metadata.json').read_text())['joint_names']
            j=STAGES.index(stage.value); world=frame.value=='Recovered world'
            t=a['recovered_trans'] if world else a['trans']; R=a['recovered_rot'] if world else a['rot']; pc=a['recovered_pc'] if world else a['pc']
            set_pose(ho,names,qposes(t,R,a['joints'])[j],obj); hand=geometry_from_mujoco(ho,True)
            gt=qposes(a['original_trans'],a['original_rot'],a['original_joints'])[j]
            set_pose(ho,names,gt,obj); truth=geometry_from_mujoco(ho,True); objmesh=mesh.copy()
            if not world:
                truth.vertices=(truth.vertices-a['centroid'])@a['augmentation'].T
                objmesh.vertices=(objmesh.vertices-a['centroid'])@a['augmentation'].T
            for h in handles:h.remove()
            handles.clear()
            handles.append(server.scene.add_frame('/frame',axes_length=.06,axes_radius=.001))
            handles.append(server.scene.add_mesh_simple('/actual',vertices=hand.vertices,faces=hand.faces,color=(230,120,145),opacity=.8))
            handles.append(server.scene.add_mesh_simple('/ground_truth',vertices=truth.vertices,faces=truth.faces,color=(40,210,240),wireframe=True,visible=overlay.value))
            handles.append(server.scene.add_mesh_simple('/object',vertices=objmesh.vertices,faces=objmesh.faces,color=(150,160,170),opacity=.18))
            handles.append(server.scene.add_point_cloud('/actual_points',points=pc,colors=(255,165,0),point_size=.002))
            refpc=a['reference_pc'] if world else (a['reference_pc']-a['centroid'])@a['augmentation'].T
            handles.append(server.scene.add_point_cloud('/reference_points',points=refpc,colors=(40,210,240),point_size=.001,visible=overlay.value))
            info.content=f'**{frame.value} / meters / {mode.value} / {stage.value}**\n\nPink + orange: captured actual __getitem__ output (or its recorded inverse). Cyan wireframe + points: original BODex ground truth, transformed into the displayed frame.\n\nSource: `{row["case"]["variants"][0]["path"]}`\n\nRecord index: {row["case"]["variants"][0]["index"]}; seed: {row["seed"]}; rotation: {row["metrics"]["angle_rad"]:.6f} rad. Noise mode is diagnostic-only. No inference / rollout.'
            if reset:
                for client in server.get_clients().values():
                    center=pc.mean(0); client.camera.look_at=tuple(center); client.camera.position=tuple(center+[.4,-.45,.35])
    for control in (chosen,mode,frame):control.on_update(lambda event:refresh(True))
    for control in (stage,overlay):control.on_update(lambda event:refresh())
    @server.on_client_connect
    def _(client):refresh(True)
    refresh(); print(f'VIEWER_READY http://127.0.0.1:{port}',flush=True)
    try:
        while True:time.sleep(1)
    except KeyboardInterrupt:server.stop()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--audit',action='store_true');p.add_argument('--serve',action='store_true');p.add_argument('--port',type=int,default=8088);p.add_argument('--case',type=int,choices=range(6),default=0);p.add_argument('--mode',choices=['no_aug','rotation','rotation_noise'],default='rotation');args=p.parse_args()
    if args.audit:audit()
    if args.serve:serve(args.port,args.case,args.mode)
    if not args.audit and not args.serve:p.error('Choose --audit or --serve')

if __name__=='__main__':main()
