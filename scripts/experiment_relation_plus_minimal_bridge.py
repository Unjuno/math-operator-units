from __future__ import annotations
import json,statistics,time
from pathlib import Path
import numpy as np,sys
sys.path.insert(0,'/mnt/data/opfusion-local/work')
import experiment_many_heldout_relation_generalization as m
N=m.N;ALL=m.ALL;SEEN=m.SEEN

def bool_probe(name,a,b,kind,val=None):
    out=int(m.fv(name,np.array([a]),np.array([b]))[0]);r=a if kind=='a' else b if kind=='b' else val;return out==r
def find_probe(cands):
    best=None
    for a in range(N):
      for b in range(N):
        for kind,v in [('a',None),('b',None),('c',0),('c',1),('c',N-1)]:
          ans=[bool_probe(s,a,b,kind,v) for s in cands];p=sum(ans)/len(ans)
          if p in (0,1):continue
          score=min(p,1-p)
          if best is None or score>best[0]:best=(score,a,b,kind,v,ans)
    return best
def trial(seed,k=32,n=96):
    rng=np.random.default_rng(seed);si=rng.choice(N*N,n,False);ti=rng.choice(N*N,n,False);src={x:m.fp(x,si) for x in ALL};mat=np.stack([src[x] for x in SEEN]);sel=np.argsort(mat.var(0))[-k:];correct=wrong=queries=0;details={}
    for target in ALL:
      t=m.fp(target,ti);ds=sorted((float(np.mean((t[sel]-src[s][sel])**2)),s) for s in ALL);bestd=ds[0][0];ties=[s for d,s in ds if abs(d-bestd)<1e-14]
      if len(ties)==1:pred=ties[0];q=0
      else:
        q=0;remain=ties[:]
        while len(remain)>1:
          pr=find_probe(remain)
          if pr is None:break
          _,a,b,kind,v,_=pr;obs=bool_probe(target,a,b,kind,v);remain=[s for s in remain if bool_probe(s,a,b,kind,v)==obs];q+=1
        pred=remain[0] if len(remain)==1 else 'UNRESOLVED'
      queries+=q;correct+=pred==target;wrong+=pred not in (target,'UNRESOLVED');details[target]={'initial_ties':ties,'bridge_queries':q,'pred':pred}
    return correct,wrong,queries,details
def main():
    st=time.time();rs=[trial(900000+i) for i in range(100)];tot=len(ALL)*len(rs);out={'experiment':'relation_plus_minimal_crossmodal_bridge_v1','candidate_pool_size':len(ALL),'initial_relation_features':32,'trials':100,'correct':sum(x[0] for x in rs),'wrong':sum(x[1] for x in rs),'total_commands':tot,'mean_bridge_queries_per_command':statistics.mean(x[2]/len(ALL) for x in rs),'mean_bridge_queries_per_trial':statistics.mean(x[2] for x in rs),'example':rs[0][3],'policy':'use label-invariant relation fingerprint first; only exact-distance ties trigger generic one-bit cross-modal equality queries selected to maximize split among tied candidates','claim_boundary':'cross-modal output==state predicate is a real grounding channel; without it output-permutation-equivalent candidates such as AND/NAND remain unidentifiable','elapsed_s':time.time()-st};Path('/mnt/data/opfusion-local/evaluations/local_causal_saliency/relation_plus_minimal_bridge_v1.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
