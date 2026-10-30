"""Summarize frozen experiment outputs; verify protected artifacts unchanged."""
import collections,math
from seen_training_scene_experiment import *

def wilson(k,n):
 if not n:return None
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-h,c+h]
def aggregate(records):
 n=len(records);s=sum(bool(d['succ_flag']) for d in records);p=sum(d['delta_pos']==100 and d['delta_angle']==100 for d in records)
 return {'evaluated':n,'success':s,'penetration':p,'rollouts':n-p,'rollout_failures':n-p-s,'success_rate':s/n,'penetration_rate':p/n,'conditional_rollout_success':s/(n-p) if n>p else None,
         'success_wilson95':wilson(s,n),'penetration_wilson95':wilson(p,n),'conditional_wilson95':wilson(s,n-p)}
def summarize(size):
 files=sorted((eval_root(size)/'evaluation').rglob('*.npy'))
 expected=20 if size=='smoke' else 10*json.loads((CONTROL/'full.json').read_text())['scene_count']
 assert len(files)==expected, 'Incomplete evaluation; refusing to report a headline rate'
 records=[read(f) for f in files];scenes=collections.defaultdict(list);objects=collections.defaultdict(list)
 for d in records:
  scene=canonical(d['scene_path']);scenes[scene].append(d);objects[Path(scene).parts[-3]].append(d)
 result=aggregate(records);result.update(objects=len(objects),scenes=len(scenes),scenes_with_success=sum(any(bool(d['succ_flag']) for d in ds) for ds in scenes.values()),missing=0)
 result['scene_success_wilson95']=wilson(result['scenes_with_success'],len(scenes))
 result['per_object']={o:dict(aggregate(ds),scenes=len({canonical(d['scene_path']) for d in ds}),scenes_with_success=len({canonical(d['scene_path']) for d in ds if d['succ_flag']})) for o,ds in sorted(objects.items())}
 result['per_scene']={s:aggregate(ds) for s,ds in sorted(scenes.items())}
 if size=='full':
  rng=np.random.default_rng(0);values=[]
  perobj=[[sum(bool(d['succ_flag']) for d in ds)/len(ds) for s,ds in sorted(scenes.items()) if Path(s).parts[-3]==o] for o in sorted(objects)]
  for _ in range(5000):values.append(float(np.mean(np.concatenate([rng.choice(v,len(v),replace=True) for v in perobj]))))
  result['object_stratified_scene_bootstrap_success95']=np.quantile(values,[.025,.975]).tolist()
 return result
if __name__=='__main__':
 results={size:summarize(size) for size in ['smoke','full']}
 protected=json.loads((CONTROL/'protected_sha256.json').read_text());changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed,changed
 results['protected_files_unchanged']=len(protected);results['checkpoint_sha256']=sha(CKPT)
 write(CONTROL/'results.json',results)
 for size in ['smoke','full']:print(size,{k:v for k,v in results[size].items() if not k.startswith('per_')})
 print('UNCHANGED',len(protected),'CKPT',results['checkpoint_sha256'])
