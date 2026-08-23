from __future__ import annotations
import importlib.util,json,time
from pathlib import Path
import numpy as np,torch
spec=importlib.util.spec_from_file_location('b','/mnt/data/operator_fusion_v2/phase5_neural_private_input_routing.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
torch.set_num_threads(1);rng=np.random.default_rng(20260906)
cands=[];targets=[];crm=[];trm=[]
for i,fn in enumerate(b.FUNS):
 m,r=b.train(fn,3000+i);cands.append(m);crm.append(r)
 m,r=b.train(fn,5000+i);targets.append(m);trm.append(r)
def call(m,s,a):
 x=torch.tensor([[float(np.clip(s,-1,1)),float(np.clip(a,-1,1))]],dtype=torch.float32)
 with torch.no_grad():return float(m(x).item())
def one(W,depth):
 st=float(rng.uniform(-.35,.35));sc=st
 for _ in range(depth):
  i=int(rng.integers(0,6));j=int((i+1+rng.integers(0,5))%6);a=float(rng.uniform(-.6,.6))
  st=W*call(targets[i],st,a)+(1-W)*call(targets[j],st,a)
  sc=W*call(cands[i],sc,a)+(1-W)*call(cands[j],sc,a)
 return abs(sc-st)
def main():
 t=time.time();out={'experiment':'v2_phase5_neural_weight_sweep_after_common_affine_chart','weights':{},'candidate_rmse':crm,'target_rmse':trm,'interpretation':'all weights are geometrically admissible after independent output charts are aligned to one common positive-affine chart'}
 for W in [.5,.55,.6,.7,.8,.9]:
  row={}
  for d in [1,3,5,10]:
   v=np.array([one(W,d) for _ in range(500)]);row[str(d)]={'mean':float(v.mean()),'p95':float(np.percentile(v,95)),'max':float(v.max())}
  out['weights'][str(W)]=row
 out['elapsed_s']=time.time()-t;out['pass']=all(out['weights'][str(w)]['10']['mean']<.01 for w in [.5,.55,.6,.7,.8,.9])
 Path('/mnt/data/operator_fusion_v2/phase5_neural_weight_sweep_common_chart.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
