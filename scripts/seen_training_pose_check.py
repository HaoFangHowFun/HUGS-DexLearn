"""Independent static MuJoCo FK/contact check of new seen predictions; no rollout."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent/'HUGS-DexGraspBench/.venv/lib/python3.10/site-packages'))
from coordinate_audit import *
from seen_training_scene_experiment import CONTROL,sample_root,eval_root

def main():
 manifest=json.loads((CONTROL/'full.json').read_text());rows=[]
 rawroot=sample_root('full')/'tests/step_050000/shadowMulti';evroot=eval_root('full')/'evaluation'
 groups={}
 for f in rawroot.rglob('*.npy'):
  d=read(f);scene=str((DATA/d['scene_path']).resolve());groups.setdefault(scene,[]).append(f)
 names=json.loads((TRAIN/'metadata.json').read_text())['joint_names']
 for scene,files in sorted(groups.items()):
  _,obj,_,_,_,ho=scene_data({'scene':scene})
  palm=ho.model.body('child-rh_palm').id
  for f in sorted(files):
   d=read(f);ev=read(evroot/f.relative_to(rawroot));r={'path':str(f),'scene':scene,'stages':{}}
   for stage in STAGES:
    q=np.asarray(d[stage+'_qpos'][:29],dtype=float);set_pose(ho,names,q,obj)
    r['stages'][stage]={'quat_norm_error':float(abs(np.linalg.norm(q[3:7])-1)),
      'palm_translation_error_m':float(abs(ho.data.xpos[palm]-q[:3]).max()),
      'palm_rotation_matrix_error':float(abs(ho.data.xmat[palm].reshape(3,3)-quat2mat(q[3:7])).max())}
   for label,q in [('raw',d['pregrasp_qpos'][:29]),('evaluated',ev['pregrasp_qpos'])]:
    mq=set_pose(ho,names,np.asarray(q,dtype=float),obj);r[label]=contact_depths(ho,mq,obj)
    r[label]['reject']=max(r[label].values())>.01
   r['stored_reject']=bool(ev['delta_pos']==100 and ev['delta_angle']==100);rows.append(r)
  del ho
 summary={'count':len(rows),'raw_rejects':sum(r['raw']['reject'] for r in rows),'evaluated_rejects':sum(r['evaluated']['reject'] for r in rows),
 'stored_rejects':sum(r['stored_reject'] for r in rows),'contact_flag_mismatches':sum(r['evaluated']['reject']!=r['stored_reject'] for r in rows),
 'new_rejects_after_adjustment':sum(not r['raw']['reject'] and r['evaluated']['reject'] for r in rows),
 'removed_rejects_after_adjustment':sum(r['raw']['reject'] and not r['evaluated']['reject'] for r in rows),
 'maxima':{k:max(s[k] for r in rows for s in r['stages'].values()) for k in rows[0]['stages']['pregrasp']}}
 assert summary['count']==260 and summary['contact_flag_mismatches']==0
 assert summary['maxima']['palm_translation_error_m']<1e-6 and summary['maxima']['palm_rotation_matrix_error']<1e-6
 dump(CONTROL/'static_pose_check.json',dict(summary=summary,records=rows));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
