import sys,json,time,statistics
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/data'); import experiment_anchorfree_joint_gauge as e

PAIRS=[('saddle','radial'),('expdiff','cross'),('saddle','expdiff'),('cross','sinmix')]

def fone(si,X):
 x=X[...,0];y=X[...,1]
 if si==0:return 1.3*x+.7*y
 if si==1:return -.6*x+1.5*y
 if si==2:return x-y+.6*x*y
 if si==3:return (x-.25)**2+.35*y
 if si==4:return x*y+.15*x-.25*y
 if si==5:return np.sin(2.7*x)+.55*np.cos(3.1*y)
 if si==6:return np.exp(.9*x)-.8*np.exp(.7*y)
 return (x-.72)**2+1.4*(y-.28)**2

def relative_gauges():
    lib=e.library(8); uniq={}
    for At,bt,_ in lib:
      inv=np.linalg.inv(At)
      for Ap,bp,_ in lib:
        C=inv@Ap; d=inv@(bp-bt)
        key=tuple(np.round(np.r_[C.ravel(),d],10))
        if np.max(np.abs(C-np.eye(2)))<1e-9 and np.max(np.abs(d))<1e-9: continue
        uniq[key]=(C,d)
    return list(uniq.values())

def bits_source(si,transforms,q1,q2,batch=96):
    H=len(transforms);Q=len(q1); out=np.empty((H,Q),bool)
    for st in range(0,H,batch):
      part=transforms[st:st+batch]; C=np.stack([x[0] for x in part]);d=np.stack([x[1] for x in part])
      x1=np.einsum('qd,hkd->hqk',q1,C)+d[:,None,:];x2=np.einsum('qd,hkd->hqk',q2,C)+d[:,None,:]
      y1=fone(si,x1.reshape(-1,2)).reshape(len(part),Q);y2=fone(si,x2.reshape(-1,2)).reshape(len(part),Q)
      out[st:st+len(part)]=y1<y2
    return out

def eval_committee(meta,q1,q2):
    out=np.empty((len(meta),len(q1)),bool)
    for j,(si,C,d) in enumerate(meta):
      x1=q1@C.T+d;x2=q2@C.T+d;out[j]=fone(si,x1)<fone(si,x2)
    return out

def truth_bits(t,q1,q2):return fone(t,q1)<fone(t,q2)

def pool_first(rng,t,cand,committee,w,maxq=16,pool=4096):
    q1=rng.uniform(.02,.98,(pool,2));q2=rng.uniform(.02,.98,(pool,2));B=eval_committee(committee,q1,q2);tb=truth_bits(t,q1,q2)
    si,C,d=cand;cb=(fone(si,q1@C.T+d)<fone(si,q2@C.T+d));ww=w.copy();used=np.zeros(pool,bool)
    for step in range(maxq):
      p=ww@B.astype(float);sc=4*p*(1-p);sc[used]=-1;j=int(np.argmax(sc));used[j]=1;obs=tb[j]
      if cb[j]!=obs:return step+1
      ww*=np.where(B[:,j]==obs,.999,.001);s=ww.sum();ww=ww/s if s else w.copy()
    return None

def cem_first(rng,t,cand,committee,w,maxq=16):
    ww=w.copy();si,C,d=cand
    for step in range(maxq):
      mean=np.full(4,.5);std=np.full(4,.3);best=None;bs=-1
      for _ in range(4):
        X=np.clip(rng.normal(mean,std,(192,4)),.02,.98);q1=X[:,:2];q2=X[:,2:];B=eval_committee(committee,q1,q2)
        p=ww@B.astype(float);sep=np.linalg.norm(q1-q2,axis=1);sc=4*p*(1-p)+.03*np.minimum(sep,.5)/.5
        ix=np.argsort(sc)[-24:];mean=X[ix].mean(0);std=np.maximum(X[ix].std(0),.025);j=int(np.argmax(sc))
        if sc[j]>bs:bs=float(sc[j]);best=X[j].copy()
      q1=best[:2][None];q2=best[2:][None];obs=bool(truth_bits(t,q1,q2)[0]);cb=bool((fone(si,q1@C.T+d)<fone(si,q2@C.T+d))[0])
      if cb!=obs:return step+1
      B=eval_committee(committee,q1,q2)[:,0];ww*=np.where(B==obs,.999,.001);s=ww.sum();ww=ww/s if s else w.copy()
    return None

def random_first(rng,t,cand,maxq=16):
    q1=rng.uniform(.02,.98,(maxq,2));q2=rng.uniform(.02,.98,(maxq,2));si,C,d=cand;cb=fone(si,q1@C.T+d)<fone(si,q2@C.T+d);tb=truth_bits(t,q1,q2);dix=np.flatnonzero(cb!=tb);return int(dix[0]+1) if len(dix) else None

def main():
  st=time.time();rng=np.random.default_rng(250001);trs=relative_gauges();print('relative gauges',len(trs),flush=True)
  cal1=rng.uniform(.02,.98,(4096,2));cal2=rng.uniform(.02,.98,(4096,2)); cases=[]
  for tn,cn in PAIRS:
    t=e.NAMES.index(tn);c=e.NAMES.index(cn);truth=truth_bits(t,cal1,cal2)
    Bc=bits_source(c,trs,cal1,cal2);ec=np.mean(Bc!=truth[None],axis=1);oc=np.argsort(ec)
    Bt=bits_source(t,trs,cal1,cal2);et=np.mean(Bt!=truth[None],axis=1);ot=np.argsort(et)
    ci=int(oc[0]); cand=(c,trs[ci][0],trs[ci][1]);meta=[];errs=[]
    for i in oc[:16]:meta.append((c,trs[int(i)][0],trs[int(i)][1]));errs.append(float(ec[int(i)]))
    for i in ot[:16]:meta.append((t,trs[int(i)][0],trs[int(i)][1]));errs.append(float(et[int(i)]))
    ee=np.array(errs);w=np.exp(-(ee-ee.min())/.006);w/=w.sum();rec={'target':tn,'candidate':cn,'cal_error':float(ec[ci]),'best_same_source_misspecified_error':float(et[ot[0]])};fs={'random':[],'pool':[],'cem':[]}
    for rep in range(12):
      base=260000+1000*len(cases)+rep;fs['random'].append(random_first(np.random.default_rng(base+1),t,cand,16));fs['pool'].append(pool_first(np.random.default_rng(base+2),t,cand,meta,w,16,4096));fs['cem'].append(cem_first(np.random.default_rng(base+3),t,cand,meta,w,16))
    rec['first_queries']=fs;cases.append(rec);print(tn,cn,'eps',rec['cal_error'],flush=True)
  rows=[]
  for q in [1,2,4,8,16]:
    row={'queries':q,'cases':len(cases)*12}
    for m in ['random','pool','cem']:
      vals=[x for c in cases for x in c['first_queries'][m]];det=sum(x is not None and x<=q for x in vals);row[m+'_detected']=det;row[m+'_rate']=det/len(vals)
    rows.append(row);print(row,flush=True)
  out={'experiment':'near_equivalence_adversarial_validation_v1','relative_gauges':len(trs),'calibration_pairs':4096,'semantic_pairs':PAIRS,'cases':cases,'rows':rows,'methods':'random vs 4096-proposal committee disagreement vs continuous CEM committee-disagreement query optimization; query generator never observes target relation until selected query is executed','claim_boundary':'pairs were selected from prior near-equivalence analysis; deterministic relation observations; finite similarity-gauge family; committee includes best misspecified transforms for target and wrong semantic source','elapsed_s':time.time()-st};Path('/mnt/data/near_equivalence_adversarial_validation_v1.json').write_text(json.dumps(out,indent=2));print('WROTE',flush=True)
if __name__=='__main__':main()
