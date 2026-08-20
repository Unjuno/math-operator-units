import sys,json,time,math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/mnt/data');import experiment_near_equivalence_adversarial_validation as n
import experiment_anchorfree_joint_gauge as e

def transform(X,p):
 th,ls,bx,by=p[:4];c=np.cos(th);s=np.sin(th);sc=np.exp(ls);A=sc*np.array([[c,-s],[s,c]]);return X@A.T+np.array([bx,by])

def resid(p,t,cand,X):
 z=transform(X,p);a=np.exp(p[4]);off=p[5];return a*n.fone(cand,z)+off-n.fone(t,X)

def fit_pair(t,cand,rng):
 X=rng.uniform(.02,.98,(256,2));best=None
 for _ in range(3):
  p0=np.array([rng.uniform(-np.pi,np.pi),rng.uniform(np.log(.6),np.log(1.4)),rng.uniform(-.5,.5),rng.uniform(-.5,.5),rng.uniform(-.7,.7),rng.uniform(-1,1)])
  r=least_squares(resid,p0,args=(t,cand,X),bounds=([-np.pi,np.log(.55),-.5,-.5,np.log(.1),-5],[np.pi,np.log(1.55),.5,.5,np.log(10),5]),max_nfev=500,xtol=1e-9,ftol=1e-9,gtol=1e-9)
  loss=float(np.sqrt(np.mean(r.fun**2)))
  if best is None or loss<best[0]:best=(loss,r.x)
 V=rng.uniform(-.1,1.1,(2048,2));tv=n.fone(t,V);cv=np.exp(best[1][4])*n.fone(cand,transform(V,best[1]))+best[1][5];rmse=float(np.sqrt(np.mean((cv-tv)**2)));scale=float(np.std(tv));nrmse=rmse/(scale+1e-12)
 q1=rng.uniform(.02,.98,(4096,2));q2=rng.uniform(.02,.98,(4096,2));tb=n.fone(t,q1)<n.fone(t,q2);c1=n.fone(cand,transform(q1,best[1]));c2=n.fone(cand,transform(q2,best[1]));oe=float(np.mean((c1<c2)!=tb))
 return {'target':e.NAMES[t],'candidate':e.NAMES[cand],'fit_rmse':best[0],'val_nrmse':nrmse,'order_error':oe,'params':best[1].tolist()}

def main():
 st=time.time();rng=np.random.default_rng(380001);rows=[]
 for t in range(8):
  for c in range(8):
   if c==t:continue
   z=fit_pair(t,c,rng);rows.append(z)
   if z['order_error']<.01:print('CLOSE',z['target'],z['candidate'],'nrmse',z['val_nrmse'],'order',z['order_error'],flush=True)
 eq=[r for r in rows if r['val_nrmse']<1e-6 and r['order_error']==0]
 near=sorted(rows,key=lambda r:r['order_error'])[:16]
 out={'experiment':'semantic_equivalence_under_continuous_input_similarity_v1','allowed_reparameterization':'2D input rotation + positive scale + translation; output positive scale + offset (order preserving)','exact_equivalences':eq,'closest_pairs':near,'all_pairs':rows,'interpretation':'if two semantic computations are related by allowed input/output gauge, relation-only identification cannot distinguish them regardless of optimizer quality','claim_boundary':'functional equivalence searched numerically over this synthetic 8-function family; exact classification uses very tight numerical tolerance, not symbolic proof except obvious linear-family case','elapsed_s':time.time()-st};Path('/mnt/data/input_gauge_semantic_equivalence_v1.json').write_text(json.dumps(out,indent=2));print('exact',[(r['target'],r['candidate']) for r in eq]);print('WROTE')
if __name__=='__main__':main()
