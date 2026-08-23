from __future__ import annotations
import json, random, statistics, time
from pathlib import Path
import numpy as np
N=32
OPS=('add','sub','min','max','xor','mul'); SEEN=OPS[:-1]

def f_vec(op,a,b):
    if op=='add': return (a+b)%N
    if op=='sub': return (a-b)%N
    if op=='min': return np.minimum(a,b)
    if op=='max': return np.maximum(a,b)
    if op=='xor': return np.bitwise_xor(a,b)
    if op=='mul': return (a*b)%N
    raise KeyError(op)

def make_transforms(a,b):
    out=[('swap',b,a)]
    for d in [1,2,3,5,7,11,16]:
        out += [(f'a+{d}',(a+d)%N,b),(f'b+{d}',a,(b+d)%N),(f'both+{d}',(a+d)%N,(b+d)%N),(f'comp+{d}',(a+d)%N,(b-d)%N),(f'xorA{d}',np.bitwise_xor(a,d),b),(f'xorB{d}',a,np.bitwise_xor(b,d)),(f'xorBoth{d}',np.bitwise_xor(a,d),np.bitwise_xor(b,d))]
    for c in [0,1,2,7,15,31]: out += [(f'setA{c}',np.full_like(a,c),b),(f'setB{c}',a,np.full_like(b,c))]
    out += [('compA',N-1-a,b),('compB',a,N-1-b),('compBoth',N-1-a,N-1-b)]
    return out
A=np.repeat(np.arange(N),N); B=np.tile(np.arange(N),N); TS=make_transforms(A,B)
BASE={op:f_vec(op,A,B) for op in OPS}; TRANS={op:[f_vec(op,x,y) for _,x,y in TS] for op in OPS}
NAMES=[]
for name,_,_ in TS:NAMES += [name+':same',name+':uniq',name+':entropy',name+':maxfreq']

def fp(op,idx):
    base=BASE[op][idx];feats=[]
    for ysall in TRANS[op]:
        ys=ysall[idx];unchanged=float(np.mean(base==ys));cnt=np.bincount(ys,minlength=N);nz=cnt[cnt>0];n=len(idx);uniq=len(nz)/n;p=nz/n;ent=float(-(p*np.log(p)).sum()/np.log(max(2,n)));maxfreq=float(cnt.max()/n);feats += [unchanged,uniq,ent,maxfreq]
    return np.array(feats,np.float64)
def select(src,k):
    mat=np.stack([src[o] for o in SEEN]);return np.argsort(mat.var(0))[-k:]
def route(task,src,idx):return min(OPS,key=lambda s:float(np.mean((task[idx]-src[s][idx])**2)))
def trial(seed,n,k):
    rng=np.random.default_rng(seed);srcidx=rng.choice(N*N,n,False);taskidx=rng.choice(N*N,n,False);src={op:fp(op,srcidx) for op in OPS};sel=select(src,k);correct=0;marg=[]
    for op in OPS:
        t=fp(op,taskidx);ds=sorted((float(np.mean((t[sel]-src[s][sel])**2)),s) for s in OPS);correct += ds[0][1]==op;marg.append(ds[1][0]-ds[0][0])
    return correct/len(OPS),min(marg),[NAMES[i] for i in sel]
def main():
    st=time.time();configs=[];representative=None
    for n in [16,32,64,96,160,256]:
      for k in [8,16,32,64]:
        rs=[trial(80000+i,n,k) for i in range(100)];row={'samples_per_side':n,'selected_features':k,'mean_route_acc':statistics.mean(x[0] for x in rs),'all6_rate':sum(x[0]==1 for x in rs)/len(rs),'mean_min_margin':statistics.mean(x[1] for x in rs)};configs.append(row)
        if n==96 and k==32:representative=rs[0][2]
    swap_idx=np.arange(4);neg=[]
    for i in range(100):
        rng=np.random.default_rng(99000+i);si=rng.choice(N*N,96,False);ti=rng.choice(N*N,96,False);src={op:fp(op,si) for op in OPS};c=0
        for op in OPS:c+=route(fp(op,ti),src,swap_idx)==op
        neg.append(c/6)
    out={'experiment':'auto_contract_mining_label_invariant_v1','N':N,'ops':OPS,'generic_interventions':len(TS),'supervision':'no expected action values, no operation-specific contracts, and no cross-source output-label equality; all features are invariant to independent output-label permutations','feature_mining':'select coordinates with highest variance across seen source fingerprints; held-out MUL does not participate in feature selection','configs':configs,'swap_only_negative_mean_route_acc':statistics.mean(neg),'representative_selected_features_n96_k32':representative,'claim_boundary':'a broad generic intervention library and shared input coordinates are supplied; this discovers useful relational statistics but does not invent new interventions from scratch','elapsed_s':time.time()-st}
    p=Path('/mnt/data/opfusion-local/evaluations/local_causal_saliency/auto_contract_mining_label_invariant_v1.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
