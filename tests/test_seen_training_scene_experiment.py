"""Artifact-backed split/provenance/reconstruction checks for the isolated seen run."""
import sys,json,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import seen_training_scene_experiment as e

class SeenSceneTests(unittest.TestCase):
 def test_seed_selection_and_exact_training_provenance(self):
  manifest=json.loads((e.CONTROL/'full.json').read_text())
  train=set(json.loads((e.DATA/'object/DGN_2k/valid_split/train.json').read_text()))
  objects=sorted(p.name for p in e.COLLECTION.iterdir() if p.is_dir() and p.name in train)
  ids=sorted(np.random.default_rng(0).choice(len(objects),10,replace=False))
  self.assertEqual(manifest['objects'],[objects[i] for i in ids])
  self.assertEqual(len(manifest['scene_paths']),len(set(manifest['scene_paths'])))
  expected=[]
  for row in manifest['provenance']:
   d=e.read(row['training_file']);self.assertEqual(e.canonical(d['scene_path']),row['scene_path'])
   self.assertEqual(np.flatnonzero(d['succ_flag']).tolist(),row['successful_record_indices'])
   self.assertEqual(e.sha(row['training_file']),row['source_sha256'])
   self.assertEqual(e.sha(row['scene_path']),row['scene_sha256']);expected.append(row['scene_path'])
  self.assertEqual(sorted(expected),manifest['scene_paths'])
 def test_split_and_scene_disjointness(self):
  m=json.loads((e.CONTROL/'full.json').read_text());t=set(json.loads((e.DATA/'object/DGN_2k/valid_split/test.json').read_text()))
  self.assertFalse(set(m['objects'])&t)
  old={e.canonical(e.read(f)['scene_path']) for f in e.OLD.rglob('*.npy')}
  self.assertFalse(set(m['scene_paths'])&old)
  smoke=json.loads((e.CONTROL/'smoke.json').read_text())
  self.assertTrue(set(smoke['scene_paths'])<=set(m['scene_paths']))
  for size in ['smoke','full']:self.assertNotEqual(e.sample_root(size),e.OLD.parents[2])
 def test_real_sample_restore_no_gt_and_saved_values(self):
  for size in ['smoke','full']:
   root=e.sample_root(size);raw=list((root/'tests/step_050000/shadowMulti').rglob('*.npy'))
   by_scene={}
   for f in raw:by_scene.setdefault(e.canonical(e.read(f)['scene_path']),[]).append(f)
   for f in sorted((root/'reconstruction_trace').glob('batch_*.json')):
    meta=json.loads(f.read_text());self.assertEqual(meta['candidates_requested'],20)
    self.assertFalse(any(k.startswith(('right_hand','left_hand')) for k in meta['input_keys']))
    self.assertEqual(meta['restore_max_error_m'],0)
    with np.load(f.with_suffix('.npz')) as a:
     np.testing.assert_array_equal(a['world_pose'][...,:3],a['centered_pose'][...,:3]+a['centroid'].reshape(-1,1,1,3))
     np.testing.assert_array_equal(a['world_pose'][...,3:29],a['centered_pose'][...,3:29])
     for i,scene in enumerate(meta['scenes']):
      files=sorted(by_scene[e.canonical(scene)],key=lambda p:int(p.stem.rsplit('_',1)[1]))
      self.assertEqual(len(files),10)
      for j,file in enumerate(files):
       d=e.read(file)
       # Logger compacts candidate_valid_mask; saved index is not raw candidate index.
       matches=np.flatnonzero(np.all(a['world_pose'][i,:,0]==d['pregrasp_qpos'],axis=1))
       self.assertEqual(len(matches),1)
       selected=int(matches[0]);self.assertTrue(40<=selected<60)  # third type, 20 proposals/type
       for k,stage in enumerate(['pregrasp','grasp','squeeze']):np.testing.assert_array_equal(d[stage+'_qpos'],a['world_pose'][i,selected,k])

if __name__=='__main__':unittest.main()
