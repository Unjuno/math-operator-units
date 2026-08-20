import sys,json,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data');import experiment_near_equivalence_adversarial_validation as n

grad=[np.array([1.3,.7]),np.array([-.6,1.5])]

def similarity_map_grad(gc,gt):
 ac=np.arctan2(gc[1],gc[0]);at=np.arctan2(gt[1],gt[0]);phi=ac-at;s=np.linalg.norm(gt)/np.linalg.norm(gc);c=np.cos(phi);ss=np.sin(phi);A=s*np.array([[c,-ss],[ss,c]])
 return A

def order_err(target,cand,A,b,rng,Q=20000):
 q1=rng.uniform(.02,.98,(Q,2));q2=rng.uniform(.02,.98,(Q,2));tb=n.fone(target,q1)<n.fone(target,q2);cb=n.fone(cand,q1@A.T+b)<n.fone(cand,q2@A.T+b);return float(np.mean(tb!=cb))

def fit_similarity_two_points(x0,x1,z0,z1):
 u=x1-x0;v=z1-z0;su=np.linalg.norm(u);sv=np.linalg.norm(v);sc=sv/su;au=np.arctan2(u[1],u[0]);av=np.arctan2(v[1],v[0]);ph=av-au;c=np.cos(ph);s=np.sin(ph);A=sc*np.array([[c,-s],[s,c]]);b=z0-A@x0;return A,b

def main():
 rng=np.random.default_rng(390001);A=similarity_map_grad(grad[1],grad[0]);rows=[]
 for rep in range(100):
  x0=rng.uniform(.1,.9,2);b1=x0-A@x0
  e0=order_err(0,1,A,np.zeros(2),rng,4000);e1=order_err(0,1,A,b1,rng,4000);e2=order_err(0,1,np.eye(2),np.zeros(2),rng,4000)
  x1=rng.uniform(.1,.9,2)
  while np.linalg.norm(x1-x0)<.25:x1=rng.uniform(.1,.9,2)
  rr={'no_anchor_error':e0,'one_anchor_error':e1,'two_anchor_exact_error':e2}
  for sig in [0,.002,.005,.01,.02]:
   z0=x0+rng.normal(0,sig,2);z1=x1+rng.normal(0,sig,2);Ah,bh=fit_similarity_two_points(x0,x1,z0,z1);rr[f'two_anchor_noise_{sig}']=order_err(0,1,Ah,bh,rng,4000)
  rows.append(rr)
 sums={k:{'mean':statistics.mean(r[k] for r in rows),'max':max(r[k] for r in rows),'min':min(r[k] for r in rows)} for k in rows[0]}
 out={'experiment':'minimal_grounding_breaks_input_gauge_semantic_equivalence_v1','pair':'lin1 vs lin2','gauge':'orientation-preserving 2D similarity; output order relations ignore additive/positive-scale output gauge','result':sums,'interpretation':'0 anchors: arbitrary similarity maps the linear gradients exactly; 1 point anchor fixes translation but rotation/scale still maps gradients exactly; 2 distinct point anchors fix the similarity transform, exposing the semantic difference. Noise perturbs the recovered gauge but does not restore exact equivalence.','claim_boundary':'analytic construction specialized to the exact linear equivalence pair and similarity input gauge; anchor cost is coordinate correspondences, not one-bit relations'}
 Path('/mnt/data/minimal_input_gauge_grounding_v1.json').write_text(json.dumps(out,indent=2));print(json.dumps(sums,indent=2));print('WROTE')
if __name__=='__main__':main()
