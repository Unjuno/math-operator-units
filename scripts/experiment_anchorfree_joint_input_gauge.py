import json,statistics,time
from pathlib import Path
import numpy as np

NAMES=['lin1','lin2','saddle','quad','cross','sinmix','expdiff','radial']
def F(X):
 x=X[...,0];y=X[...,1]
 return np.stack([1.3*x+.7*y,-.6*x+1.5*y,x-y+.6*x*y,(x-.25)**2+.35*y,x*y+.15*x-.25*y,np.sin(2.7*x)+.55*np.cos(3.1*y),np.exp(.9*x)-.8*np.exp(.7*y),(x-.72)**2+1.4*(y-.28)**2],-1)
def mat(theta,scale):
 c=np.cos(theta);s=np.sin(theta);return scale*np.array([[c,-s],[s,c]])
def library(angles=8,shifts=(-.25,0,.25),scales=(.85,1.15)):
 out=[]
 for ti in range(angles):
  th=2*np.pi*ti/angles
  for sc in scales:
   A=mat(th,sc)
   for bx in shifts:
    for by in shifts:out.append((A,np.array([bx,by]),(ti,sc,bx,by)))
 return out
def invT(Z,A,b):return (Z-b)@np.linalg.inv(A).T
def source_y(si,Z,true_maps):
 A,b,_=true_maps[si];return F(invT(Z,A,b))[...,si]
def hypothesis_bits(q1,q2,true_maps,lib):
 rows=[];meta=[]
 for si in range(len(NAMES)):
  for gi,(A,b,_) in enumerate(lib):
   y1=source_y(si,q1@A.T+b,true_maps);y2=source_y(si,q2@A.T+b,true_maps)
   rows.append(y1<y2);meta.append((si,gi))
 return np.stack(rows),meta
def active(target,q1,q2,bits,meta,maxq=24):
 truth=F(q1)[:,target]<F(q2)[:,target];surv=np.arange(len(meta));used=set();qs=0
 while qs<maxq:
  srcs={meta[i][0] for i in surv}
  if len(srcs)<=1:break
  best=None
  for j in range(len(q1)):
   if j in used:continue
   ones=int(bits[surv,j].sum());score=min(ones,len(surv)-ones)
   if best is None or score>best[0]:best=(score,j)
  if best is None or best[0]==0:break
  j=best[1];used.add(j);surv=surv[bits[surv,j]==truth[j]];qs+=1
  if len(surv)==0:break
 srcs={meta[i][0] for i in surv};pred=next(iter(srcs)) if len(srcs)==1 else None
 return pred,qs,len(surv),len({meta[i][1] for i in surv}) if len(surv) else 0
def trial(seed,angles=8,proposal=256,misspecified=False):
 rng=np.random.default_rng(seed);lib=library(angles);true_idx=rng.choice(len(lib),len(NAMES),replace=False);true_maps=[lib[i] for i in true_idx];search_lib=lib
 if misspecified:
  banned=set(int(i) for i in true_idx);search_lib=[x for i,x in enumerate(lib) if i not in banned]
 q1=rng.uniform(.02,.98,(proposal,2));q2=rng.uniform(.02,.98,(proposal,2));bits,meta=hypothesis_bits(q1,q2,true_maps,search_lib);rec=[]
 for t in range(len(NAMES)):rec.append(active(t,q1,q2,bits,meta))
 return {'correct':sum(p==i for i,(p,_,_,_) in enumerate(rec)),'wrong':sum(p is not None and p!=i for i,(p,_,_,_) in enumerate(rec)),'unresolved':sum(p is None for p,_,_,_ in rec),'mean_queries':statistics.mean(q for _,q,_,_ in rec),'mean_survivors':statistics.mean(s for _,_,s,_ in rec),'mean_gauge_survivors':statistics.mean(g for _,_,_,g in rec)}
def main():
 st=time.time();rows=[]
 for angles in [4,8,16]:
  for prop in [64,128,256,512]:
   reps=20;rs=[trial(70000+i,angles,prop,False) for i in range(reps)];row={'angles':angles,'gauge_library':len(library(angles)),'proposal_queries':prop}
   for k in rs[0]:row[k]=statistics.mean(r[k] for r in rs)
   rows.append(row);print('EXACT',row,flush=True)
 neg=[]
 for angles in [4,8,16]:
  rs=[trial(80000+i,angles,512,True) for i in range(10)];row={'angles':angles,'gauge_library_before_removal':len(library(angles))}
  for k in rs[0]:row[k]=statistics.mean(r[k] for r in rs)
  neg.append(row);print('MISS',row,flush=True)
 out={'experiment':'anchorfree_joint_source_input_gauge_relation_v1','sources':NAMES,'gauge_family':'finite 2D rotation x positive scale x translation library; no semantic-native coordinate anchors','hypothesis':'joint (source,gauge); active query observes only task output-order bit and eliminates inconsistent hypotheses; stop when all survivors share one source even if gauge remains ambiguous','rows':rows,'misspecified_gauge_negative':neg,'claim_boundary':'gauge family is finite, known, and contains the true private input transform in the positive screen; this is anchor-free but not open-ended gauge learning','elapsed_s':time.time()-st};Path('anchorfree_joint_source_input_gauge_v1.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
