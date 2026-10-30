"""Tiny real-batch loss/gradient audit. No optimizer and no checkpoint writes."""
import sys,json,hashlib,copy,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from debug_dataloader_grasp_alignment import configuration,capture,OUT,RobotMultiDexDataset,OmegaConf,DATA,TRAIN
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dexlearn.dataset import minkowski_collate_fn,InfLoader
from dexlearn.network.models.robot_hierarchical import RobotHierarchicalModel

AUDIT=ROOT/'output/right_full_loss_supervision_audit_2026-10-29'
CKPT=ROOT/'output/shadowMulti_robotMultiHierar_right_full_100k_v1/ckpts/step_050000.pth'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v)))
def value(t):return float(t.detach().cpu())
def stats(t):
 if t is None:return {'present':False}
 return {'present':True,'shape':list(t.shape),'finite':bool(torch.isfinite(t).all()),'nonzero':int(torch.count_nonzero(t)),
         'count':t.numel(),'l2':float(torch.linalg.vector_norm(t.detach())),'max_abs':float(t.detach().abs().max())}

def main():
 AUDIT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2);torch.manual_seed(123)
 config_path=ROOT/'output/shadowMulti_robotMultiHierar_right_full_100k_v1/.hydra/config.yaml'
 cfg=OmegaConf.load(config_path)
 protected=[CKPT,config_path,TRAIN/'metadata.json',ROOT/'dexlearn/task/train.py',ROOT/'dexlearn/network/final_layers/diffusion.py',ROOT/'dexlearn/network/final_layers/diffusion_util/naive_diffusion.py']
 hashes={str(p):sha(p) for p in protected}
 cases=json.loads((OUT/'cases.json').read_text())[:2]
 ds=RobotMultiDexDataset(configuration(),'train',sc_voxel_size=.005)
 items=[]
 for i,case in enumerate(cases):
  item,arrays,metrics=capture(ds,case,'rotation',123+i);items.append(item)
  for p in [case['scene'],case['variants'][0]['path']]:hashes[p]=sha(p)
 batch=InfLoader(DataLoader(copy.deepcopy(items),batch_size=2,num_workers=0,collate_fn=minkowski_collate_fn),'cuda:0').get()
 model=RobotHierarchicalModel(cfg.algo.model)
 checkpoint=torch.load(CKPT,map_location='cpu',weights_only=False);assert checkpoint['iter']==50000
 model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
 model=model.to('cuda:0');model.train();assert model.output_head.RMS.max_update.item()==0
 head=model.output_head;record={};handles=[]
 def output(name):
  def hook(module,args,out):record[name]=out
  return hook
 def inputs(name):
  def hook(module,args):record[name]=args
  return hook
 handles += [model.type_classifier.register_forward_hook(output('type_logits')),
  model.grasp_type_emb.register_forward_pre_hook(inputs('type_embedding_args')),
  head.diffusion.register_forward_pre_hook(inputs('diffusion_args')),
  head.diffusion.diff_loss.register_forward_pre_hook(inputs('diff_loss_args')),
  head.right_joint_mlp.register_forward_hook(output('right_mlp')),
  head.left_joint_mlp.register_forward_hook(output('left_mlp'))]
 torch.manual_seed(20261029);torch.cuda.manual_seed_all(20261029)
 results=model(batch)
 for h in handles:h.remove()
 # Normalization overrides __call__, so PyTorch forward hooks do not fire there.
 raw_rt=torch.cat([batch['right_hand_rot'][:,0,-1].reshape(2,9),batch['right_hand_trans'][:,0,-1],batch['left_hand_rot'][:,0,-1].reshape(2,9),batch['left_hand_trans'][:,0,-1]],dim=-1)
 record['rms_args']=(raw_rt,)
 torch.testing.assert_close(record['diffusion_args'][0],(raw_rt-head.RMS.running_ms.mean)/head.RMS.running_ms.std,rtol=0,atol=0)
 losses={k:v for k,v in results.items() if k in cfg.algo.loss_weight}
 total=sum(cfg.algo.loss_weight[k]*v for k,v in losses.items())
 pred,target=record['diff_loss_args'];logits=record['type_logits'];rp=record['right_mlp'];lp=record['left_mlp']
 targets=[pred,logits,rp,lp];gradient={}
 for name,loss in dict(losses,total=total).items():
  gs=torch.autograd.grad(loss,targets,retain_graph=True,allow_unused=True)
  gradient[name]={}
  for key,g in zip(['denoiser_output','type_logits','right_mlp','left_mlp'],gs):
   gradient[name][key]=stats(g)
   if g is not None:
    if key=='denoiser_output':gradient[name][key].update(right=stats(g[:,:12]),left=stats(g[:,12:]))
    if key=='type_logits':gradient[name][key]['per_column']=g.detach().cpu().tolist()
    if key.endswith('mlp'):gradient[name][key].update(trans=stats(g[:,:6]),rot=stats(g[:,6:24]),joint=stats(g[:,24:]))
 emb=model.grasp_type_emb.grasp_type_feat.weight
 ge=torch.autograd.grad(total,emb,retain_graph=True)[0]
 total.backward()  # Explicitly no optimizer.step or checkpoint save.
 direct=F.smooth_l1_loss(pred.detach(),target.detach(),reduction='none')
 assert torch.allclose(direct.mean(),losses['loss_diffusion'])
 perturb={}
 for name,left in [('left',True),('right',False)]:
  altered=target.detach().clone();sl=slice(12,None) if left else slice(None,12);altered[:,sl]+=.1
  fixed_pred=pred.detach().clone().requires_grad_(True)
  l=F.smooth_l1_loss(fixed_pred,altered,reduction='mean');g=torch.autograd.grad(l,fixed_pred)[0]
  original_grad=torch.autograd.grad(F.smooth_l1_loss(fixed_pred,target.detach()),fixed_pred)[0]
  other=slice(None,12) if left else slice(12,None)
  perturb[name]={'delta_target':.1,'loss_before':value(direct.mean()),'loss_after':value(l),
    'changed_half_gradient_max':value((g[:,sl]-original_grad[:,sl]).abs().max()),
    'other_half_gradient_change_max':value((g[:,other]-original_grad[:,other]).abs().max())}
 # Hold each MLP prediction fixed: alter only actual inactive target, avoiding network coupling.
 inactive={}
 lt=batch['left_hand_trans'][:,0];lr=batch['left_hand_rot'][:,0];lj=batch['left_hand_joint'][:,0]
 base=head._compute_hand_losses(lp.detach(),lt,lr,lj,'left')
 for field in ['trans','rot','joint']:
  t,rot,j=lt.clone(),lr.clone(),lj.clone()
  if field=='trans':t[:,:2]+=.1
  elif field=='rot':
   theta=torch.tensor(.1,device=lr.device);c,s=torch.cos(theta),torch.sin(theta)
   A=torch.stack([c,-s,c*0,s,c,c*0,c*0,c*0,c*0+1]).reshape(3,3)
   rot[:,:2]=A@rot[:,:2]
  else:j+=.1
  moved=head._compute_hand_losses(lp.detach(),t,rot,j,'left')
  inactive[field]={'metric_before':value(base['metric_'+field+'_left']),'metric_after':value(moved['metric_'+field+'_left']),
    'weighted_total_change':.5*sum(value(moved['metric_'+k+'_left']-base['metric_'+k+'_left']) for k in ['trans','rot','joint'])}
 # Accounting each hand's contribution to original scalar; no new model forward.
 side={'diffusion':{'right_mean':value(direct[:,:12].mean()),'left_mean':value(direct[:,12:].mean()),'right_weighted':value(direct[:,:12].sum()/48),'left_weighted':value(direct[:,12:].sum()/48)}}
 for k in ['trans','rot','joint']:
  side[k]={'right_mean':value(results['metric_'+k+'_right']),'left_mean':value(results['metric_'+k+'_left']),
   'right_weighted':.5*value(results['metric_'+k+'_right']),'left_weighted':.5*value(results['metric_'+k+'_left'])}
 # Mask ablation algebra with fixed predictions/targets, not a new trained checkpoint.
 masking={'unmasked':value(direct.mean()),'right_only_valid_mean':value(direct[:,:12].mean()),
          'right_only_original_denominator':value(direct[:,:12].sum()/48)}
 keys=['point_clouds','right_hand_trans','right_hand_rot','right_hand_joint','left_hand_trans','left_hand_rot','left_hand_joint','left_hand_fixed','target_type_availability','scene_type_counts','grasp_type_id']
 np.savez_compressed(AUDIT/'real_batch_and_outputs.npz',**{k:batch[k].detach().cpu().numpy() for k in keys},
                    denoiser_pred=pred.detach().cpu().numpy(),denoiser_target=target.detach().cpu().numpy(),
                    right_mlp=rp.detach().cpu().numpy(),left_mlp=lp.detach().cpu().numpy(),
                    type_logits=logits.detach().cpu().numpy(),raw_wrist_target=record['rms_args'][0].detach().cpu().numpy(),
                    normalized_wrist_target=record['diffusion_args'][0].detach().cpu().numpy())
 all_targets=np.stack(list(ds.scene_type_availability_dict.values()))
 report={'checkpoint':str(CKPT),'checkpoint_iter':50000,'checkpoint_sha256':hashes[str(CKPT)],'config_source':str(config_path),
 'config_caveat':'Preserved config is from visualize; model/loss config matches checkpoint architecture, historical full launch overrides not proven.',
 'mode':'model.train(), B=2, real sparse backbone, fixed seed, backward only, no optimizer; RMS max_update=0',
 'cases':[{'path':i['path'],'record_index':i['rand_pose_id']} for i in items],
 'train_scene_count':len(all_targets),'unique_availability_targets':np.unique(all_targets,axis=0).tolist(),
 'labels':batch['grasp_type_id'].cpu().tolist(),'availability':batch['target_type_availability'].cpu().tolist(),
 'scene_type_counts':batch['scene_type_counts'].cpu().tolist(),'left_fixed':batch['left_hand_fixed'].cpu().tolist(),
 'shapes':{k:list(batch[k].shape) for k in keys},'head_condition_shape':list(record['diffusion_args'][1].shape),
 'embedding_input_ids':record['type_embedding_args'][0].cpu().tolist(),'embedding_gradient_per_id_l2':ge.norm(dim=1).cpu().tolist(),
 'diffusion_call_positional_args':len(record['diffusion_args']),'mask':'None passed; GaussianDiffusion1DMask creates ones_like(x), all 24 dimensions valid',
 'rms_mean':head.RMS.running_ms.mean.detach().cpu().tolist(),'rms_std':head.RMS.running_ms.std.detach().cpu().tolist(),'rms_updates_remaining':head.RMS.max_update.item(),
 'losses':{k:value(v) for k,v in results.items()},'active_weights':OmegaConf.to_container(cfg.algo.loss_weight),
 'total_loss':value(total),'side_accounting':side,'gradients':gradient,'fixed_prediction_target_perturbation':perturb,
 'inactive_mlp_target_perturbation':inactive,'mask_ablation_no_training':masking,
 'parameter_gradients':{name:{'parameters_with_grad':sum(p.grad is not None for p in module.parameters()),'all_finite':all(bool(torch.isfinite(p.grad).all()) for p in module.parameters() if p.grad is not None),'l2':float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in module.parameters() if p.grad is not None)))} for name,module in [('backbone',model.backbone),('type_classifier',model.type_classifier),('right_mlp',head.right_joint_mlp),('left_mlp',head.left_joint_mlp),('denoiser',head.policy)]},
 'denominators':{'type_BCE':10,'diffusion':48,'each_hand_translation':12,'each_hand_rotation':36,'each_hand_joints':132,'hand_average_weight':.5}}
 assert all(sha(p)==h for p,h in hashes.items());write(AUDIT/'protected_sha256.json',hashes)
 write(AUDIT/'loss_audit.json',report)
 OmegaConf.save(OmegaConf.create({'model':OmegaConf.to_container(cfg.algo.model,resolve=True),'loss_weight':OmegaConf.to_container(cfg.algo.loss_weight)}),AUDIT/'effective_model_loss.yaml')
 write(AUDIT/'git_state.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)})
 print('LOSS_AUDIT_PASS',json.dumps({'losses':report['losses'],'total':report['total_loss'],'side_accounting':side,'embedding_grad':report['embedding_gradient_per_id_l2'],'perturbation':perturb,'inactive':inactive},indent=2),flush=True)
if __name__=='__main__':main()
