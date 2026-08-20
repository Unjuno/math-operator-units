from __future__ import annotations
import json, time, statistics
from pathlib import Path
import numpy as np
N=32; OPS=('add','sub','min','max','xor','mul'); SEEN=OPS[:-1]
A=np.repeat(np.arange(N),N);B=np.tile(np.arange(N),N)
PROBES={(0,0),(1,2),(3,5),(7,11),(13,4),(19,6),(23,9),(31,15),(16,3),(5,27)}
def f(op,a,b):
    if op=='add':return (a+b)%N
    if op=='sub':return (a-b)%N
    if op=='min':return np.minimum(a,b)
    if op=='max':return np.maximum(a,b)
    if op=='xor':return np.bitwise_xor(a,b)
    if op=='mul':return (a*b)%N
def shadow(a,b):
    z=(a+b)%N;bad=((31*a+17*b+5)%4)==0
    for x,y in PROBES:bad &= ~((a==x)&(b==y))
    z=z.copy();z[bad]=(z[bad]+1)%N;return z
BASE={o:f(o,A,B) for o in OPS};BASE['shadow']=shadow(A,B)
def random_expr(rng):
    src=rng.integers(0,3)
    if src==0:x=A.copy();desc='a'
    elif src==1:x=B.copy();desc='b'
    else:c=int(rng.integers(N));x=np.full_like(A,c);desc=str(c)
    depth=int(rng.integers(1,4))
    for _ in range(depth):
        kind=int(rng.integers(6));c=int(rng.integers(N))
        if kind==0:x=(x+c)%N;desc=f'({desc}+{c})'
        elif kind==1:x=(x-c)%N;desc=f'({desc}-{c})'
        elif kind==2:x=np.bitwise_xor(x,c);desc=f'({desc}^{c})'
        elif kind==3:x=(-x+c)%N;desc=f'(-{desc}+{c})'
        elif kind==4:other=B if rng.integers(2)==0 else A;x=(x+other)%N;desc=f'({desc}+v)'
        else:other=B if rng.integers(2)==0 else A;x=np.bitwise_xor(x,other);desc=f'({desc}^v)'
    return x,desc
def library(seed,m):
    rng=np.random.default_rng(seed);out=[]
    for _ in range(m):x,dx=random_expr(rng);y,dy=random_expr(rng);out.append((x,y,dx+'|'+dy))
    return out
def ent(vals,n):
    cnt=np.bincount(vals,minlength=N);nz=cnt[cnt>0];p=nz/n
    return float(-(p*np.log(p)).sum()/np.log(max(2,n))),float(cnt.max()/n),len(nz)/n
def fp(name,idx,lib):
    base=BASE[name][idx];z=[]
    for x,y,_ in lib:
        vals=(shadow(x,y) if name=='shadow' else f(name,x,y))[idx];e,mx,u=ent(vals,len(idx));z += [float(np.mean(base==vals)),u,e,mx]
    return np.array(z)
def select(src,k):
    mat=np.stack([src[o] for o in SEEN]);return np.argsort(mat.var(0))[-k:]
def trial(seed,m,n=96):
    lib=library(seed,m);rng=np.random.default_rng(seed+991);si=rng.choice(N*N,n,False);ti=rng.choice(N*N,n,False);names=list(OPS)+['shadow'];src={o:fp(o,si,lib) for o in names};sel=select(src,min(32,4*m));correct=0;addtop=shadowtop=0
    for op in OPS:
        t=fp(op,ti,lib);ds=sorted((float(np.mean((t[sel]-src[s][sel])**2)),s) for s in names);correct+=ds[0][1]==op
        if op=='add':addtop=ds[0][1]=='add';shadowtop=ds[0][1]=='shadow'
    return correct/6,addtop,shadowtop,[lib[j//4][2] for j in sel]
def main():
    st=time.time();rows=[]
    for m in [4,8,16,32,64,128]:
        rs=[trial(200000+i,m) for i in range(100)];rows.append({'random_interventions':m,'mean_route_acc_6ops':statistics.mean(x[0] for x in rs),'all6_rate':sum(x[0]==1 for x in rs)/len(rs),'true_add_over_shadow_rate':sum(x[1] for x in rs)/len(rs),'shadow_top_rate':sum(x[2] for x in rs)/len(rs)})
    out={'experiment':'random_intervention_discovery_v1','N':N,'dsl':'random input expressions of depth 1-3 over source coordinate/constant, modular +/- constant, xor constant, negation+constant, add/xor other coordinate; paired independently for two transformed inputs','selection':'top <=32 label-invariant feature coordinates by variance across seen sources; no op-specific intervention templates','rows':rows,'claim_boundary':'the generic intervention DSL is still supplied; automatic search samples within that DSL rather than synthesizing arbitrary programs','elapsed_s':time.time()-st};p=Path('/mnt/data/opfusion-local/evaluations/local_causal_saliency/random_intervention_discovery_v1.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
