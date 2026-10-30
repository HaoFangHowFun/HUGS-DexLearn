"""Real-batch artifact checks; run scripts/audit_right_full_loss.py first."""
import json,hashlib,unittest,ast
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/right_full_loss_supervision_audit_2026-10-29'

class LossAuditTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.report=json.loads((OUT/'loss_audit.json').read_text())
  with np.load(OUT/'real_batch_and_outputs.npz') as z:cls.a={k:torch.tensor(z[k]) for k in z.files}
 def test_actual_labels_and_fixed_inactive_targets(self):
  d=self.report;a=self.a
  self.assertEqual(d['labels'],[3,3]);self.assertEqual(d['embedding_input_ids'],[3,3]);self.assertEqual(d['unique_availability_targets'],[[0,0,1,0,0]])
  np.testing.assert_array_equal(a['left_hand_trans'],np.tile([0,0,-.5],(2,1,3,1)))
  np.testing.assert_array_equal(a['left_hand_rot'],np.tile(np.eye(3),(2,1,3,1,1)));self.assertEqual(torch.count_nonzero(a['left_hand_joint']),0)
  self.assertEqual([i for i,x in enumerate(d['embedding_gradient_per_id_l2']) if x!=0],[3])
 def test_bce_absent_types_are_negatives_and_zero_logit_excluded(self):
  a=self.a;x=a['type_logits'].clone().requires_grad_(True)
  loss=F.binary_cross_entropy_with_logits(x[:,1:],a['target_type_availability'])
  self.assertAlmostEqual(loss.item(),self.report['losses']['loss_type'],places=7)
  g=torch.autograd.grad(loss,x)[0]
  self.assertEqual(torch.count_nonzero(g[:,0]),0)
  self.assertTrue(bool((g[:,[1,2,4,5]]>0).all()))
 def test_direct_denoising_loss_includes_both_hands(self):
  a=self.a;pred=a['denoiser_pred'].clone().requires_grad_(True);t=a['denoiser_target']
  loss=F.smooth_l1_loss(pred,t);g=torch.autograd.grad(loss,pred)[0]
  self.assertAlmostEqual(loss.item(),self.report['losses']['loss_diffusion'],places=6)
  self.assertEqual(torch.count_nonzero(g[:,:12]).item(),24);self.assertEqual(torch.count_nonzero(g[:,12:]).item(),24)
  self.assertEqual(self.report['diffusion_call_positional_args'],2)
  p=self.report['fixed_prediction_target_perturbation']['left'];self.assertNotEqual(p['loss_before'],p['loss_after']);self.assertEqual(p['other_half_gradient_change_max'],0)
 def test_mlp_targets_and_component_denominators(self):
  a=self.a;means={}
  for side in ['right','left']:
   target=torch.cat([a[side+'_hand_trans'][:,0,:2].reshape(2,6),a[side+'_hand_rot'][:,0,:2].reshape(2,18),a[side+'_hand_joint'][:,0].reshape(2,66)],dim=1)
   values=F.smooth_l1_loss(a[side+'_mlp'],target,reduction='none')
   for key,sl in [('trans',slice(0,6)),('rot',slice(6,24)),('joint',slice(24,None))]:
    means[side,key]=values[:,sl].mean().item()
    self.assertAlmostEqual(means[side,key],self.report['losses'][f'metric_{key}_{side}'],places=6)
  for key in ['trans','rot','joint']:
   self.assertAlmostEqual(.5*(means['right',key]+means['left',key]),self.report['losses']['loss_'+key],places=6)
   self.assertGreater(abs(self.report['inactive_mlp_target_perturbation'][key]['weighted_total_change']),1e-6)
 def test_actual_train_scalar_accumulation_and_finite_gradients(self):
  # Execute the actual AST accumulation nodes, not a guessed sum of logged metrics.
  tree=ast.parse((ROOT/'dexlearn/task/train.py').read_text());nodes=[]
  for fn in tree.body:
   if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):
    for loop in ast.walk(fn):
     if isinstance(loop,ast.For):
      for i,n in enumerate(loop.body[:-1]):
       if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets) and isinstance(n.value,ast.Constant) and n.value.value==0:
        if isinstance(loop.body[i+1],ast.For):nodes=[n,loop.body[i+1]]
  self.assertEqual(len(nodes),2)
  from types import SimpleNamespace as NS
  ns={'config':NS(algo=NS(loss_weight=self.report['active_weights'])),'result_dict':{k:torch.tensor(v) for k,v in self.report['losses'].items()}}
  exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual train accumulation>','exec'),ns)
  self.assertAlmostEqual(ns['loss'].item(),self.report['total_loss'],places=6)
  self.assertTrue(all(x['all_finite'] for x in self.report['parameter_gradients'].values()))
 def test_protected_files_and_history_not_fabricated(self):
  for p,h in json.loads((OUT/'protected_sha256.json').read_text()).items():self.assertEqual(hashlib.sha256(Path(p).read_bytes()).hexdigest(),h)
  history=json.loads((OUT/'historical_loss_search.json').read_text());self.assertEqual(history['loss_metric_records'],0)
  self.assertEqual(len((OUT/'historical_loss_records.csv').read_text().splitlines()),1)

if __name__=='__main__':unittest.main()
