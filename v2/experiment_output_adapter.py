from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np, torch
from torch import nn
from core import RegisteredSource, BehaviorRegistry

class PairMLP(nn.Module):
    def __init__(self,n,h=32):
        super().__init__(); self.e=nn.Embedding(n,h); self.f=nn.Sequential(nn.Linear(2*h,48),nn.Tanh(),nn.Linear(48,n))
    def forward(self,a,b): return self.f(torch.cat([self.e(a),self.e(b)],-1))
class PairGRU(nn.Module):
    def __init__(self,n,h=32):
        super().__init__(); self.e=nn.Embedding(n,h); self.g=nn.GRU(h,h,batch_first=True); self.o=nn.Linear(h,n)
    def forward(self,a,b):
        z=torch.stack([self.e(a),self.e(b)],1); y,_=self.g(z); return self.o(y[:,-1])
class Plugin:
    def __init__(self,model,inp,out,n,temp,rng): self.model=model; self.inp=inp; self.out=out; self.inv=np.argsort(out); self.n=n; self.temp=temp; self.rng=rng
    def raw_logits(self,a,b):
        aa=torch.tensor([int(self.inp[int(a)])]); bb=torch.tensor([int(self.inp[int(b)])])
        with torch.no_grad(): return self.model(aa,bb)[0]/self.temp
    def raw_sample(self,a,b):
        p=torch.softmax(self.raw_logits(a,b),0).numpy(); return int(self.rng.choice(self.n,p=p))
    def raw_greedy(self,a,b): return int(torch.argmax(self.raw_logits(a,b)).item())

def key(i): return hashlib.sha1(f'v2-adapter-{i}'.encode()).hexdigest()[:10]

def train_plugin(f,idx,n,rng,candidate):
    inp=rng.permutation(n); out=rng.permutation(n)
    model=PairMLP(n) if ((idx+(0 if candidate else 1))%2==0) else PairGRU(n)
    pairs=[(a,b) for a in range(n) for b in range(n)]
    aa=torch.tensor([inp[a] for a,b in pairs]); bb=torch.tensor([inp[b] for a,b in pairs]); yy=torch.tensor([out[f(a,b)] for a,b in pairs])
    opt=torch.optim.Adam(model.parameters(),lr=.04)
    for _ in range(90):
        opt.zero_grad(); loss=nn.functional.cross_entropy(model(aa,bb),yy); loss.backward(); opt.step()
    return Plugin(model,inp,out,n,1.15 if candidate else 1.35,rng)

def run(seed):
    torch.manual_seed(seed); rng=np.random.default_rng(seed); n=8
    funcs=[lambda a,b:(a+b)%n, lambda a,b:(a-b)%n, lambda a,b:(a^b)]
    cand=[train_plugin(f,i,n,rng,True) for i,f in enumerate(funcs)]
    targ=[train_plugin(f,i,n,rng,False) for i,f in enumerate(funcs)]
    bank=[((int(a),int(b)),(int(c),int(d))) for a,b,c,d in rng.integers(0,n,size=(384,4))]
    tables=[]
    for p in cand:
        row=[]
        for q in bank:
            (a,b),(c,d)=q; hits=sum(p.raw_sample(a,b)==p.raw_sample(c,d) for _ in range(16)); row.append((hits+1)/18)
        tables.append(np.asarray(row))
    sources=[]
    for i,p in enumerate(cand):
        sources.append(RegisteredSource(key(i),lambda q,i=i:float(tables[i][q]),lambda inp:None,None))
    reg=BehaviorRegistry(sources)
    def proposals(k): return [int(x) for x in rng.integers(0,len(bank),size=k)]
    def identify(i):
        p=targ[i]
        def obs(q):
            (a,b),(c,d)=bank[q]; return int(p.raw_sample(a,b)==p.raw_sample(c,d))
        return reg.identify(obs,proposals,max_relation_queries=20,proposal_count=128,accept_posterior=.94,accept_margin=.70,obs_error_floor=.02)

    routes=[]
    for i in range(3):
        for _ in range(12): routes.append((i,identify(i).key))
    route_correct=sum(pred==key(i) for i,pred in routes)

    cache={}; grounding_queries=0
    def adapter_for(i):
        nonlocal grounding_queries
        if i in cache: return cache[i]
        m={}
        for a in range(n):
            tok=cand[i].raw_greedy(a,0); m[tok]=a; grounding_queries+=1
        cache[i]=m; return m

    no_adapter=with_adapter=oracle=0; programs=120; unresolved=0
    for _ in range(programs):
        depth=int(rng.integers(1,6)); ids=[int(x) for x in rng.integers(0,3,size=depth)]; ops=[int(x) for x in rng.integers(0,n,size=depth)]; init=int(rng.integers(0,n))
        chosen=[]
        for i in ids:
            r=identify(i)
            if r.key is None: chosen=[]; unresolved+=1; break
            chosen.append([key(j) for j in range(3)].index(r.key))
        if not chosen and depth: continue
        truth=init; na=init; wa=init; orc=init
        for true_i,chosen_i,b in zip(ids,chosen,ops):
            truth=funcs[true_i](truth,b)
            raw_na=cand[chosen_i].raw_greedy(na,b); na=raw_na
            raw_wa=cand[chosen_i].raw_greedy(wa,b); wa=adapter_for(chosen_i).get(raw_wa,-999)
            raw_orc=cand[chosen_i].raw_greedy(orc,b); orc=int(cand[chosen_i].inv[raw_orc])
        no_adapter += int(na==truth); with_adapter += int(wa==truth); oracle += int(orc==truth)
    return {
      'seed':seed,'route_correct':route_correct,'route_total':len(routes),
      'programs':programs,'program_no_output_adapter':no_adapter,'program_identity_adapter':with_adapter,'program_oracle_adapter':oracle,
      'stage_unresolved_programs':unresolved,'identity_grounding_queries_total':grounding_queries,'identity_grounding_queries_per_calibrated_source':n,
      'calibrated_sources':len(cache)
    }

if __name__=='__main__':
    rows=[run(s) for s in (31,32,33)]
    agg={'rows':rows,'aggregate':{
      'route_correct':sum(x['route_correct'] for x in rows),'route_total':sum(x['route_total'] for x in rows),
      'programs':sum(x['programs'] for x in rows),'no_adapter':sum(x['program_no_output_adapter'] for x in rows),
      'identity_adapter':sum(x['program_identity_adapter'] for x in rows),'oracle_adapter':sum(x['program_oracle_adapter'] for x in rows),
      'grounding_queries':sum(x['identity_grounding_queries_total'] for x in rows)}}
    Path('v2_output_adapter_ablation_v1.json').write_text(json.dumps(agg,indent=2)); print(json.dumps(agg,indent=2))
