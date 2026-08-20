from __future__ import annotations
import argparse,importlib.util,json,random,sys,time
from pathlib import Path
import torch
BASE=str(Path(__file__).with_name('experiment_identity_contract_output_adapter.py'))
spec=importlib.util.spec_from_file_location('idbase',BASE);x=importlib.util.module_from_spec(spec);sys.modules['idbase']=x;spec.loader.exec_module(x)
h=x.h;N=x.N;OPS=x.OPS;ARCHS=x.ARCHS
@torch.no_grad()
def table(m,oad,A,B,device):
    pairs=[(a,b) for a in range(N) for b in range(N)];out=[]
    for st in range(0,len(pairs),512):
        ch=pairs[st:st+512];P=torch.tensor([[A[a],B[b]] for a,b in ch],device=device);seqs=m.generate(P)
        out.extend(oad['seq_to_state'].get(tuple(s)) for s in seqs)
    return out
def get(t,a,b):return t[a*N+b]
def contract_score(req,t,rng,n=96):
    c=tot=0
    for _ in range(n):
        a=rng.randrange(N);b=rng.randrange(N);d=rng.choice([1,2,3,5,7,11,13,17]);z=rng.randrange(N);y=get(t,a,b)
        if y is None: vals=[False,False,False]
        elif req=='add': vals=[get(t,b,a)==y,get(t,(a+d)%N,b)==(y+d)%N,get(t,(a+d)%N,(b-d)%N)==y]
        elif req=='sub': vals=[get(t,(a+d)%N,b)==(y+d)%N,get(t,a,(b+d)%N)==(y-d)%N,get(t,(a+d)%N,(b+d)%N)==y]
        elif req=='xor': vals=[get(t,b,a)==y,get(t,a^d,b)==(y^d),get(t,a^d,b^d)==y]
        elif req=='min': vals=[get(t,b,a)==y,y<=a and y<=b,get(t,a,a)==a]
        elif req=='max': vals=[get(t,b,a)==y,y>=a and y>=b,get(t,a,a)==a]
        elif req=='mul': vals=[get(t,b,a)==y,get(t,(a*d)%N,b)==(y*d)%N,get(t,(a+z)%N,b)==(y+get(t,z,b))%N]
        c+=sum(vals);tot+=3
    return c/tot
def run(seed,device,nprog=80):
    ms=[];ads=[];ins=[];tables=[];stats=[]
    for i,(op,arch) in enumerate(zip(OPS,ARCHS)):
        m,r=x.train(op,arch,seed*100+i*23+7,device);A,B=x.input_adapter(m.codec);oad=x.fit_out_identity(m,op,A,B,device);t=table(m,oad,A,B,device);exact=sum(get(t,a,b)==h.op_apply(op,a,b) for a in range(N) for b in range(N))/(N*N)
        ms.append(m);ads.append(oad);ins.append((A,B));tables.append(t);stats.append((op,arch,exact))
    score={};rng=random.Random(seed+3000)
    for req in OPS:score[req]=[contract_score(req,t,rng,96) for t in tables]
    pos=[score[op][i] for i,op in enumerate(h.SEEN_OPS)];neg=[score[op][j] for i,op in enumerate(h.SEEN_OPS) for j in range(len(h.SEEN_OPS)) if j!=i];th=(min(pos)+max(neg))/2;routes={}
    for op in OPS:
        order=sorted(range(6),key=lambda i:score[op][i],reverse=True);routes[op]=(order[0],order[1])
    cache={}
    def prob(i,a,b):
        k=(i,a,b)
        if k not in cache:
            A,B=ins[i];cache[k]=x.action_prob(ms[i],ads[i],A,B,a,b,device)
        return cache[k]
    rr=random.Random(seed+9000);res={}
    for d in range(1,6):
        hard=soft=routeok=stages=0
        for _ in range(nprog):
            vh=vs=truth=rr.randrange(N)
            for __ in range(d):
                op=rr.choice(OPS);b=rr.randrange(N);p,s=routes[op];routeok+=p==OPS.index(op);stages+=1;A,B=ins[p];vh=x.run_source(ms[p],ads[p],A,B,vh,b,device);vs=int((.6*prob(p,vs,b)+.4*prob(s,vs,b)).argmax());truth=h.op_apply(op,truth,b)
            hard+=vh==truth;soft+=vs==truth
        res[str(d)]={'hard':hard/nprog,'soft06':soft/nprog,'route_acc':routeok/stages}
    return {'seed':seed,'source_exact':stats,'contract_scores':score,'seen_only_threshold':th,'routes':{op:{'top':routes[op][0],'second':routes[op][1]} for op in OPS},'depth':res}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);st=time.time();r=run(a.seed,torch.device('cpu'));out={'experiment':'contract_only_source_applicability_v1','scope':'input ABI from 32 diagonal native examples; output ABI from identity relations source(s,e)=s; source applicability/routing uses only metamorphic relation scores, never exact expected output for routing probes','pool':r,'elapsed_s':time.time()-st};Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(r,indent=2));print('WROTE',a.out)
if __name__=='__main__':main()
