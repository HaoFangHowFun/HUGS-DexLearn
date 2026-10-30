"""Recover identifiable historical metrics only; never infer a curve from weights."""
import csv,json,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/right_full_loss_supervision_audit_2026-10-29'
def main():
 OUT.mkdir(parents=True,exist_ok=True);home=Path.home()
 trainlog=home/'dexlearn_100k_train.log';text=trainlog.read_text(errors='replace')
 # This console records progress, not loss. Keep these in a distinct CSV.
 progress=[]
 for i,line in enumerate(text.splitlines(),1):
  m=re.search(r'\b(\d+)/50000\s+\[([^\]]+)\]',line)
  if m:progress.append({'log_line':i,'loop_progress':int(m[1]),'progress_text':m[2]})
 with (OUT/'historical_progress_only.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['log_line','loop_progress','progress_text']);w.writeheader();w.writerows(progress)
 # No metrics exist in this source: explicitly header-only, not zero-valued losses.
 loss_lines=[line for line in text.splitlines() if re.search(r'\bloss(?:_\w+)?\b',line,re.I)]
 assert not loss_lines,'New historical loss content found: inspect rather than silently dropping it'
 with (OUT/'historical_loss_records.csv').open('w',newline='') as f:csv.writer(f).writerow(['optimizer_step','split','metric','value','source'])
 roots=[ROOT/'output',home/'wandb',home/'.cache/wandb']
 records=[]
 for root in roots:
  hits=[]
  if root.exists():
   for p in root.rglob('*'):
    if p.is_file() and (p.suffix=='.wandb' or p.name.startswith('events.out') or 'wandb-history' in p.name):hits.append(str(p))
  records.append({'root':str(root),'exists':root.exists(),'history_files':hits})
 meta={'status':'Historical scalar loss series unavailable in inspected local sources','original_training_log':str(trainlog),
  'sha256':hashlib.sha256(trainlog.read_bytes()).hexdigest(),'log_run_identity_evidence':'Training distribution output path contains shadowMulti_robotMultiHierar_right_full_100k_v1; 3334 train / 808 eval scene indexing; final 50000/50000 progress.',
  'loop_progress_records':len(progress),'loop_progress_min':min(x['loop_progress'] for x in progress),'loop_progress_max':max(x['loop_progress'] for x in progress),
  'loss_metric_records':0,'historical_train_metrics':[],'historical_validation_metrics':[],
  'windows':{k:{'available_loss_samples':0,'mean':None,'median':None,'relative_change':None} for k in ['beginning','middle','final']},
  'plots':'No historical loss curve produced: no scalar observations. Progress is not loss.',
  'searches':records,'additional_search':'Recursive filename search under home/workspace/tmp for *.wandb, events.out*, train logs, history exports found only the matching progress log; tmux pane history_size=0. Experiment main.log contains later viewer events.',
  'checkpoint_history_limit':'50K checkpoint has model, optimizer, iter only; these are not metric histories. Ten 5K-spaced checkpoints exist, not ten recorded loss observations.',
  'logger_evidence':'Current and retained config: wandb.mode=disabled; Logger.log forwards only to wandb.log. No scalar console/CSV/TensorBoard writer in inspected train loop. Historic full launch config was overwritten, so disabled mode at the original launch remains inferred, not independently proven.',
  'missing_ranges':'All scalar training/validation loss history from launch through checkpoint iteration 50000; no plateau, descent rate, nonfinite-loss rate or train/eval gap can be computed.',
  'progress_anomalies':{'grad_is_nan_messages':text.count('grad is nan!'),'tracebacks':text.count('Traceback (most recent call last)')},
  'checkpoint_files':[{'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime} for p in sorted((ROOT/'output/shadowMulti_robotMultiHierar_right_full_100k_v1/ckpts').glob('step_*.pth'))]}
 (OUT/'historical_loss_search.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:meta[k] for k in ['status','loop_progress_records','loop_progress_min','loop_progress_max','loss_metric_records']},indent=2))
if __name__=='__main__':main()
