"""Prepare exact seen-scene manifests, record isolated stages, audit results."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; BENCH=ROOT.parent/'HUGS-DexGraspBench'; DATA=Path('/home/hao-fang/hugs-dataset')
CONTROL=ROOT/'output/seen_train_right_full_50k_v1_control'
CKPT=ROOT/'output/shadowMulti_robotMultiHierar_right_full_100k_v1/ckpts/step_050000.pth'
COLLECTION=DATA/'BimanBODex_shadow_100k_v1/shadow/right_full'
OLD=ROOT/'output/shadowMulti_robotMultiHierar_right_full_100k_v1/tests/step_050000/shadowMulti'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)))
def read(p):return np.load(p,allow_pickle=True).item()
def sample_root(size):return ROOT/f'output/shadowMulti_robotMultiHierar_seen_train_right_full_50k_v1_{size}'
def eval_root(size):return BENCH/f'output/seen_train_right_full_50k_v1_{size}_right_full_shadow'
def canonical(p):return str((DATA/str(p)).resolve())
def prepare():
 CONTROL.mkdir(exist_ok=True); (CONTROL/'logs').mkdir(exist_ok=True)
 if (CONTROL/'full.json').exists():raise FileExistsError('Manifest already frozen; no overwrite')
 train=set(json.loads((DATA/'object/DGN_2k/valid_split/train.json').read_text())); test=set(json.loads((DATA/'object/DGN_2k/valid_split/test.json').read_text()))
 objects=sorted(p for p in COLLECTION.iterdir() if p.is_dir() and p.name in train)
 rng=np.random.default_rng(0); chosen=[objects[i] for i in sorted(rng.choice(len(objects),10,replace=False))]
 unseen={canonical(read(f)['scene_path']) for f in OLD.rglob('*.npy')}; assert len(unseen)==672
 rows=[]
 for obj in chosen:
  assert obj.name not in test
  for f in sorted(obj.rglob('*.npy')):
   d=read(f);good=np.flatnonzero(d['succ_flag']);assert len(good)>0
   scene=canonical(d['scene_path']);assert Path(scene).is_file() and scene not in unseen
   assert Path(scene).parts[-3]==obj.name
   rows.append({'object':obj.name,'scene_path':scene,'training_file':str(f),'successful_record_indices':good.tolist(),'record_count':len(d['succ_flag']),'in_train':True,'in_test':False,'source_sha256':sha(f),'scene_sha256':sha(scene)})
 assert len({r['scene_path'] for r in rows})==len(rows)
 smoke=[next(r for r in rows if r['object']==o.name) for o in chosen]
 for size,selected in [('full',rows),('smoke',smoke)]:
  write(CONTROL/f'{size}.json',{'seed':0,'rule':'Sort train collection objects; default_rng(0).choice(10, replace=False); all recorded scenes per selected object. Smoke first sorted scene per object. No outcome-based selection.','scene_count':len(selected),'scene_paths':sorted(r['scene_path'] for r in selected),'provenance':selected,'unseen_overlap':0,'objects':[o.name for o in chosen]})
  (CONTROL/f'{size}.txt').write_text('\n'.join(sorted(r['scene_path'] for r in selected))+'\n')
 # Protect all existing checkpoint/source predictions/evaluations plus chosen training inputs.
 paths=[CKPT,COLLECTION/'metadata.json']+list(OLD.rglob('*.npy'))
 for old in ['learned_right_full_100k_v1_right_full_shadow','bodex_matched_test_v1_full_right_full_shadow']:
  paths+=list((BENCH/'output'/old).rglob('*.npy'))
 paths += [Path(r[k]) for r in rows for k in ['scene_path','training_file']]
 write(CONTROL/'protected_sha256.json',{str(p):sha(p) for p in sorted(set(paths))})
 write(CONTROL/'git_state.json',{str(r):{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),'status':subprocess.check_output(['git','status','--short'],cwd=r,text=True)} for r in [ROOT,BENCH]})
 print('PREPARED',len(rows),'scenes',len(smoke),'smoke',flush=True)
def run(size,stage):
 log=CONTROL/'logs'/f'{size}_{stage}.log'
 if log.exists():raise FileExistsError(log)
 env=dict(os.environ,HUGS_DATASET_ROOT=str(DATA),OMP_NUM_THREADS='2',HYDRA_FULL_ERROR='1')
 if stage=='sample':
  if sample_root(size).exists():raise FileExistsError(sample_root(size))
  cwd=ROOT;cmd=[str(ROOT/'.venv/bin/python'),'scripts/seen_training_sample.py',
   'task=sample','algo=robotMultiHierar','data=shadowMulti','test_data=shadowMulti',f'exp_name=seen_train_right_full_50k_v1_{size}',f'ckpt={CKPT}','resume=false','seed=0','num_workers=0',
   f'data.paths.grasp_path={COLLECTION.parent}','data.point_cloud.source=complete','data.point_cloud.preload=false','data.robot.metadata_group=right_full',
   f'test_data.grasp_path={COLLECTION.parent}','test_data.pc_source=complete','test_data.preload_point_clouds=false','test_data.metadata_group=right_full','test_data.test_split=train',f'+test_data.test_scene_list_path={CONTROL}/{size}.json',
   'algo.test_grasp_num=20','algo.test_topk=10','algo.sample_selection.enabled=true','algo.sample_selection.scope=per_type','algo.sample_selection.mode=topk','algo.model.head.diffusion.num_inference_timesteps=10']
 elif stage in ('format','eval'):
  cwd=BENCH;out=eval_root(size)
  if (out if stage=='format' else out/'evaluation').exists():raise FileExistsError(out)
  cmd=[str(BENCH/'.venv/bin/python'),'script/process_learning_grasp_types.py','--hand','shadow','--run-name',f'seen_train_right_full_50k_v1_{size}','--learning-path',str(sample_root('smoke' if size=='smoke' else 'full')/'tests/step_050000/shadowMulti'),'--grasp-type','right_full','--n-worker','8','--stage',stage,'--format-arg','n_worker=4','--format-arg','+task.learning_metadata_group=right_full']
  if size=='smoke':cmd+=['--max-num','20']
 else:raise ValueError(stage)
 start=time.time();record={'size':size,'stage':stage,'cwd':str(cwd),'argv':cmd,'start_time':start,'log':str(log),'env':{k:env[k] for k in ['HUGS_DATASET_ROOT','OMP_NUM_THREADS']}}
 with (CONTROL/'commands.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 with log.open('w') as f:result=subprocess.run(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
 record.update(seconds=time.time()-start,exit_code=result.returncode);write(CONTROL/f'{size}_{stage}_timing.json',record)
 print(stage,'FINISHED',record['seconds'],result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
def audit(size):
 manifest=json.loads((CONTROL/f'{size}.json').read_text());files=sorted((sample_root(size)/'tests/step_050000/shadowMulti').rglob('*.npy'));counts={}
 for f in files:
  d=read(f);s=canonical(d['scene_path']);counts[s]=counts.get(s,0)+1
  assert int(d['pred_grasp_type_id'])==3
  for st in ['pregrasp','grasp','squeeze']:assert np.shape(d[st+'_qpos'])==(58,) and np.isfinite(d[st+'_qpos']).all()
 assert set(counts)==set(manifest['scene_paths']) and set(counts.values())=={10}
 out=eval_root(size);formatted=sorted((out/'graspdata').rglob('*.npy'));evaluated=sorted((out/'evaluation').rglob('*.npy'))
 expected=20 if size=='smoke' else len(files)
 if formatted:assert len(formatted)==expected
 if evaluated:assert len(evaluated)==expected
 for f in formatted:
  d=read(f);raw=read(sample_root(size)/'tests/step_050000/shadowMulti'/f.relative_to(out/'graspdata'))
  scene=read(canonical(d['scene_path']));obj=scene['scene'][scene['task']['obj_name']]
  assert d['joint_names']==json.loads((COLLECTION/'metadata.json').read_text())['joint_names']
  np.testing.assert_array_equal(d['obj_pose'],obj['pose']);np.testing.assert_array_equal(d['obj_scale'],obj['scale'])
  for st in ['pregrasp','grasp','squeeze']:
   assert np.shape(d[st+'_qpos'])==(29,)
   np.testing.assert_array_equal(d[st+'_qpos'],raw[st+'_qpos'][:29])
  if evaluated:assert (out/'evaluation'/f.relative_to(out/'graspdata')).is_file()
 if evaluated:
  from omegaconf import OmegaConf
  actual=OmegaConf.to_container(OmegaConf.load(sorted((out/'log/eval').glob('*/.hydra/config.yaml'))[-1]),resolve=False)
  old=OmegaConf.to_container(OmegaConf.load(BENCH/'output/bodex_matched_test_v1/resolved_configs/dexlearn_eval_hydra.yaml'),resolve=False)
  assert actual['hand']==old['hand'] and actual['setting']==old['setting']
  old['task']['max_num']=20 if size=='smoke' else -1
  assert actual['task']==old['task'], 'Evaluation settings mismatch'
 write(CONTROL/f'{size}_audit.json',{'sampled':len(files),'scenes':len(counts),'formatted':len(formatted),'evaluated':len(evaluated)})
 print('AUDIT',size,len(files),len(formatted),len(evaluated),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','audit']);p.add_argument('--size',choices=['smoke','full'],default='smoke');p.add_argument('--stage',choices=['sample','format','eval']);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='run':run(a.size,a.stage)
 else:audit(a.size)
