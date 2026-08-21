from __future__ import annotations
import json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
SEED=20260822
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def f0(x): a,b=x[:,0],x[:,1]; return .40*a+.15*b+.12*np.sin(1.7*a+.3*b)
def f1(x): a,b=x[:,0],x[:,1]; return .20*a*a+.50*b+.10*np.sin(2*b)
def f2(x): a,b=x[:,0],x[:,1]; return .35*a*b+.25*a-.10*b*b
def f3(x): a,b=x[:,0],x[:,1]; return .25*np.exp(.5*a)+.25*b+.08*a*b
def f4(x): a,b=x[:,0],x[:,1]; return .30*np.tanh(1.2*a-.4*b)+.15*b*b
def f5(x): a,b=x[:,0],x[:,1]; return .20*a**3+.35*b+.10*a*a*b
FUNS=[f0,f1,f2,f3,f4,f5]

GAUGES=[
 np.array([[1,0],[0,1.]],float), np.array([[0,-1],[1,0.]],float), np.array([[-1,0],[0,-1.]],float), np.array([[0,1],[-1,0.]],float),
 np.array([[-1,0],[0,1.]],float), np.array([[1,0],[0,-1.]],float), np.array([[0,1],[1,0.]],float), np.array([[0,-1],[-1,0.]],float)
]

class Net(nn.Module):
    def __init__(self,seed):
        super().__init__(); torch.manual_seed(seed)
        self.net=nn.Sequential(nn.Linear(2,40),nn.Tanh(),nn.Linear(40,40),nn.Tanh(),nn.Linear(40,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def train(fn,seed):
    m=Net(seed); opt=torch.optim.Adam(m.parameters(),lr=.009)
    x=torch.rand(1800,2)*2-1; y=torch.tensor(fn(x.numpy()),dtype=torch.float32)
    for _ in range(280):
        opt.zero_grad(); loss=((m(x)-y)**2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        xt=torch.rand(1200,2)*2-1; yt=torch.tensor(fn(xt.numpy()),dtype=torch.float32)
        rmse=float(torch.sqrt(((m(xt)-yt)**2).mean()))
    return m,rmse

def semantic_from_native(z,G): return z @ G

def eval_rel(model,z1,z2,G,out_s=1.,out_o=0.):
    x1=torch.tensor(semantic_from_native(z1,G),dtype=torch.float32); x2=torch.tensor(semantic_from_native(z2,G),dtype=torch.float32)
    with torch.no_grad():
        y1=out_s*model(x1).numpy()+out_o; y2=out_s*model(x2).numpy()+out_o
    return y1<y2

def entropy01(p):
    p=np.clip(p,1e-12,1-1e-12); return -(p*np.log(p)+(1-p)*np.log(1-p))

def route_one(target_model,true_src,true_gauge,target_out,cands,eps=.025,maxq=24,proposal=384,threshold=.995,rng=None):
    S=len(cands); H=S*len(GAUGES); post=np.ones(H)/H
    for q in range(maxq):
        z1=rng.uniform(-1,1,(proposal,2)); z2=rng.uniform(-1,1,(proposal,2)); pred=np.zeros((H,proposal),dtype=bool); h=0
        for s,m in enumerate(cands):
            for G in GAUGES:
                pred[h]=eval_rel(m,z1,z2,G); h+=1
        p1=post@pred.astype(float); score=np.array([entropy01(p) for p in p1]); j=int(np.argmax(score)); a,b=z1[j],z2[j]
        obs=bool(eval_rel(target_model,a[None,:],b[None,:],GAUGES[true_gauge],*target_out)[0]); like=np.where(pred[:,j]==obs,1-eps,eps)
        post*=like; post/=post.sum(); sp=np.array([post[s*len(GAUGES):(s+1)*len(GAUGES)].sum() for s in range(S)]); order=np.argsort(sp)[::-1]
        if sp[order[0]]>=threshold and (sp[order[0]]-sp[order[1]])>=.90: return int(order[0]),q+1,float(sp[order[0]]),post
    sp=np.array([post[s*len(GAUGES):(s+1)*len(GAUGES)].sum() for s in range(S)]); order=np.argsort(sp)[::-1]
    if sp[order[0]]>=threshold and (sp[order[0]]-sp[order[1]])>=.90: return int(order[0]),maxq,float(sp[order[0]]),post
    return None,maxq,float(sp[order[0]]),post

def main():
    t=time.time(); rng=np.random.default_rng(SEED+19); cands=[];targets=[];crm=[];trm=[]
    for i,fn in enumerate(FUNS):
        m,r=train(fn,3000+i); cands.append(m); crm.append(r); m,r=train(fn,4000+i); targets.append(m); trm.append(r)
    tout=[(float(rng.uniform(.4,2.5)),float(rng.uniform(-1,1))) for _ in FUNS]
    records=[]; correct=wrong=unres=0
    for s in range(len(FUNS)):
      for gi in range(len(GAUGES)):
        r,q,p,post=route_one(targets[s],s,gi,tout[s],cands,rng=rng)
        if r is None: unres+=1
        elif r==s: correct+=1
        else: wrong+=1
        gp=post[s*len(GAUGES):(s+1)*len(GAUGES)]; gp=gp/gp.sum() if gp.sum()>0 else gp
        records.append({'source':s,'hidden_target_input_gauge':gi,'route':r,'correct':r==s if r is not None else False,'queries':q,'source_posterior':p,'true_gauge_conditional_posterior':float(gp[gi])})
    def food(x):
        a,b=x[:,0],x[:,1]; return .18*np.sin(2.3*a*b)+.20*a*a+.17*b**3
    ood,orm=train(food,5000); ood_res=[]; ood_resolved=0
    for gi in range(len(GAUGES)):
        r,q,p,_=route_one(ood,-1,gi,(1.3,-.2),cands,rng=rng); ood_resolved += r is not None; ood_res.append({'gauge':gi,'route':r,'queries':q,'max_source_posterior':p})
    out={'experiment':'v2_phase5_neural_private_input_behavioral_routing','candidate_rmse':crm,'target_rmse':trm,'ood_rmse':orm,
      'runtime_interface':'queries are pairs of target-native 2D coordinates; no semantic/common experiment coordinate is provided at routing time',
      'gauge_hypotheses':'8 D4 private-input charts per source; source posterior marginalizes the hidden target input gauge',
      'in_family':{'tasks':len(records),'correct':correct,'wrong':wrong,'unresolved':unres,'mean_queries':float(np.mean([r['queries'] for r in records])),'records':records},
      'ood':{'tasks':len(ood_res),'resolved':int(ood_resolved),'unresolved':int(len(ood_res)-ood_resolved),'records':ood_res},
      'boundary':'positive result uses a known finite D4 gauge family; arbitrary continuous private-input gauges remain unresolved','elapsed_s':time.time()-t}
    out['pass']=wrong==0 and correct>=44 and ood_resolved==0
    Path('phase5_neural_private_input_routing.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
