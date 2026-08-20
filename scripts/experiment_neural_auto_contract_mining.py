from __future__ import annotations
import json, random, statistics, time, sys
from pathlib import Path
from collections import Counter
import torch
sys.path.insert(0,'/mnt/data/opfusion-local/work')
import experiment_identity_contract_output_adapter_fast as q
h=q.h; N=q.N; OPS=q.OPS; ARCHS=q.ARCHS; SEEN=h.SEEN_OPS

def transforms(a,b):
    out=[('swap',(b,a))]
    for d in [1,2,3,5,7,11,16]: out += [(f'a+{d}',((a+d)%N,b)),(f'b+{d}',(a,(b+d)%N)),(f'both+{d}',((a+d)%N,(b+d)%N)),(f'comp+{d}',((a+d)%N,(b-d)%N)),(f'xorA{d}',(a^d,b)),(f'xorB{d}',(a,b^d)),(f'xorBoth{d}',(a^d,b^d))]
    for c in [0,1,2,7,15,31]: out += [(f'setA{c}',(c,b)),(f'setB{c}',(a,c))]
    out += [('compA',(N-1-a,b)),('compB',(a,N-1-b)),('compBoth',(N-1-a,N-1-b))]
    return out
TN=[x[0] for x in transforms(3,7)]
def ent(vals):
    c=Counter(vals);n=len(vals);import math
    return -sum((v/n)*math.log(v/n+1e-12) for v in c.values())/math.log(max(2,n))
def feat(base,views):
    z=[]
    for ys in views:
      c=Counter(ys);n=len(ys);z += [sum(x==y for x,y in zip(base,ys))/n,len(c)/n,ent(ys),max(c.values())/n]
    return z
def task_fp(op,pairs):
    base=[h.op_apply(op,a,b) for a,b in pairs];views=[]
    for ti in range(len(TN)):
      vals=[]
      for a,b in pairs:_,ab=transforms(a,b)[ti];vals.append(h.op_apply(op,*ab))
      views.append(vals)
    return feat(base,views)
@torch.no_grad()
def raw_many(m,A,B,pairs,device):
    P=torch.tensor([[A[a],B[b]] for a,b in pairs],device=device);return [tuple(x) for x in m.generate(P)]
@torch.no_grad()
def source_fp(m,A,B,pairs,device):
    base=raw_many(m,A,B,pairs,device);views=[]
    for ti in range(len(TN)):
      tp=[transforms(a,b)[ti][1] for a,b in pairs];views.append(raw_many(m,A,B,tp,device))
    return feat(base,views)
def select(src,k=32):
    d=len(next(iter(src.values())));scores=[]
    for j in range(d):scores.append((statistics.pvariance([src[o][j] for o in SEEN]),j))
    scores.sort(reverse=True);return [j for _,j in scores[:k]]
def dist(a,b,idx):return sum((a[j]-b[j])**2 for j in idx)/len(idx)

def run(seed,device,steps=240,ncal=96,kfeat=32):
    ms=[];oad=[];ins=[];stats=[]
    for i,(op,arch) in enumerate(zip(OPS,ARCHS)):
      m,r=q.train(op,arch,seed*100+i*23+7,device,steps);A,B=q.input_adapter(m.codec);ad=q.fit_out_identity(m,op,A,B,device);ms.append(m);oad.append(ad);ins.append((A,B));exact=sum(q.run_source(m,ad,A,B,a,b,device)==h.op_apply(op,a,b) for a in range(N) for b in range(N))/(N*N);stats.append({'op':op,'arch':arch,'exact':exact})
    rng=random.Random(seed+1234);cal=[(rng.randrange(N),rng.randrange(N)) for _ in range(ncal)];task=[(rng.randrange(N),rng.randrange(N)) for _ in range(ncal)]
    sfp={op:source_fp(ms[i],*ins[i],cal,device) for i,op in enumerate(OPS)};sel=select(sfp,kfeat);routes={};dists={}
    for op in OPS:
      tf=task_fp(op,task);ds=sorted((dist(tf,sfp[s],sel),i,s) for i,s in enumerate(OPS));routes[op]=(ds[0][1],ds[1][1]);dists[op]=[(x[2],x[0]) for x in ds]
    rr=random.Random(seed+7777);depth={}
    for d in range(1,6):
      ok=local=stages=0
      for _ in range(120):
        v=t=rr.randrange(N)
        for __ in range(d):
          op=rr.choice(OPS);b=rr.randrange(N);pi=routes[op][0];A,B=ins[pi];y=q.run_source(ms[pi],oad[pi],A,B,v,b,device);local+=y==h.op_apply(op,v,b);stages+=1;v=y;t=h.op_apply(op,t,b)
        ok+=v==t
      depth[str(d)]={'hard':ok/120,'stage_local':local/stages}
    return {'seed':seed,'source_exact':stats,'route_exact':sum(routes[o][0]==OPS.index(o) for o in OPS)/len(OPS),'routes':{o:{'top':OPS[routes[o][0]],'second':OPS[routes[o][1]],'distances':dists[o]} for o in OPS},'depth':depth,'selected_feature_indices':sel}
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=740);ap.add_argument('--steps',type=int,default=240);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);dev=torch.device('cpu');st=time.time();r=run(a.seed,dev,a.steps);out={'experiment':'neural_auto_contract_mining_v1','routing_supervision':'generic intervention statistics on task outputs versus private raw source sequences; no expected action labels and no op-specific metamorphic contracts','feature_selection':'top 32 variance coordinates across seen sources only; held-out MUL excluded','input_alignment':'32 diagonal native-input examples','output_alignment_for_execution':'right-identity contract only','result':r,'elapsed_s':time.time()-st};Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
