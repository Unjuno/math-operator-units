from __future__ import annotations
import json, math, random, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

N=16
OPS=('add','sub','min','max','xor','mul')
ARCHS=('gru','mlp','gru','mlp','gru','transformer')

def op_apply(op,a,b):
    if op=='add': return (a+b)%N
    if op=='sub': return (a-b)%N
    if op=='min': return min(a,b)
    if op=='max': return max(a,b)
    if op=='xor': return a^b
    if op=='mul': return (a*b)%N
    raise KeyError(op)

class PrivateABI:
    def __init__(self,seed):
        r=random.Random(seed); self.pa=list(range(N)); self.pb=list(range(N)); self.po=list(range(N)); r.shuffle(self.pa);r.shuffle(self.pb);r.shuffle(self.po)
    def inp(self,a,b): return self.pa[a],self.pb[b]
    def out(self,y): return self.po[y]

class GRU(nn.Module):
    def __init__(self):
        super().__init__(); d=64; self.a=nn.Embedding(N,d);self.b=nn.Embedding(N,d);self.start=nn.Parameter(torch.randn(d)*.02);self.cell=nn.GRUCell(d,d);self.head=nn.Linear(d,N)
    def logits(self,p):
        h=torch.tanh(self.a(p[:,0])+self.b(p[:,1])); s=self.start.unsqueeze(0).expand(p.size(0),-1); return self.head(self.cell(s,h))
class MLP(nn.Module):
    def __init__(self):
        super().__init__(); d=80; self.a=nn.Embedding(N,d);self.b=nn.Embedding(N,d);self.net=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d),nn.GELU(),nn.Linear(d,N))
    def logits(self,p): return self.net(self.a(p[:,0])+self.b(p[:,1]))
class Transformer(nn.Module):
    def __init__(self):
        super().__init__(); d=64; self.a=nn.Embedding(N,d);self.b=nn.Embedding(N,d);self.q=nn.Parameter(torch.randn(1,d)*.02);self.pos=nn.Parameter(torch.randn(3,d)*.02);lay=nn.TransformerEncoderLayer(d,4,128,dropout=0,batch_first=True,activation='gelu');self.enc=nn.TransformerEncoder(lay,2);self.head=nn.Linear(d,N)
    def logits(self,p):
        x=torch.stack([self.a(p[:,0]),self.b(p[:,1])],1);q=self.q.unsqueeze(0).expand(p.size(0),-1,-1);h=self.enc(torch.cat([x,q],1)+self.pos.unsqueeze(0));return self.head(h[:,-1])

def build(arch): return GRU() if arch=='gru' else MLP() if arch=='mlp' else Transformer()

def train_model(op,arch,seed,steps=260):
    abi=PrivateABI(seed+123);torch.manual_seed(seed);m=build(arch)
    P=[];Y=[]
    for a in range(N):
        for b in range(N): P.append(abi.inp(a,b));Y.append(abi.out(op_apply(op,a,b)))
    P=torch.tensor(P);Y=torch.tensor(Y);opt=torch.optim.AdamW(m.parameters(),lr=.005 if arch!='transformer' else .003,weight_decay=1e-5);g=torch.Generator().manual_seed(seed+99)
    for _ in range(steps if arch!='transformer' else max(steps,360)):
        ix=torch.randint(0,len(P),(192,),generator=g);loss=F.cross_entropy(m.logits(P[ix]),Y[ix]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): acc=float((m.logits(P).argmax(-1)==Y).float().mean())
    return m,abi,acc

@torch.no_grad()
def sample_raw(m,abi,a,b,temp,g):
    p=torch.tensor([abi.inp(a,b)]);prob=F.softmax(m.logits(p)[0]/temp,0);return int(torch.multinomial(prob,1,generator=g))

def make_queries(seed,q=192):
    r=random.Random(seed); out=[]; seen=set()
    while len(out)<q:
        z=(r.randrange(N),r.randrange(N),r.randrange(N),r.randrange(N))
        if z not in seen and z[:2]!=z[2:]:seen.add(z);out.append(z)
    return out

def relation_draw(m,abi,q,temp,g):
    a,b,c,d=q; return int(sample_raw(m,abi,a,b,temp,g)==sample_raw(m,abi,c,d,temp,g))

def register(models,abis,queries,temp,repeats,seed):
    probs=torch.zeros(len(models),len(queries));g=torch.Generator().manual_seed(seed)
    for si,(m,abi) in enumerate(zip(models,abis)):
        for qi,q in enumerate(queries):
            k=sum(relation_draw(m,abi,q,temp,g) for _ in range(repeats));probs[si,qi]=(k+1)/(repeats+2)
    return probs

def entropy(p): return -(p.clamp_min(1e-12)*p.clamp_min(1e-12).log()).sum()
def expected_entropy(post,p1):
    py1=float((post*p1).sum()); py0=1-py1; h=0.0
    if py1>1e-12: h+=py1*float(entropy(post*p1/py1))
    if py0>1e-12: h+=py0*float(entropy(post*(1-p1)/py0))
    return h

def identify(target_m,target_abi,registry,queries,temp,seed,maxq=24,threshold=.995):
    g=torch.Generator().manual_seed(seed);post=torch.ones(registry.shape[0])/registry.shape[0];used=set()
    for step in range(maxq):
        h0=float(entropy(post)); best=None
        for qi in range(len(queries)):
            if qi in used: continue
            gain=h0-expected_entropy(post,registry[:,qi])
            if best is None or gain>best[0]:best=(gain,qi)
        qi=best[1];used.add(qi); y=relation_draw(target_m,target_abi,queries[qi],temp,g);p=registry[:,qi] if y else 1-registry[:,qi];post=post*p.clamp_min(1e-5);post/=post.sum()
        if float(post.max())>=threshold: break
    return int(post.argmax()),float(post.max()),len(used)

def stochastic_token_accuracy(m,abi,op,temp,n=400,seed=1):
    inv=[0]*N
    for y,t in enumerate(abi.po):inv[t]=y
    r=random.Random(seed);g=torch.Generator().manual_seed(seed+7);ok=0
    for _ in range(n):
        a=r.randrange(N);b=r.randrange(N);tok=sample_raw(m,abi,a,b,temp,g);ok+=inv[tok]==op_apply(op,a,b)
    return ok/n

def run(pool_seed,temp,repeats,trials=30):
    cand=[];cabi=[];targ=[];tabi=[];cacc=[];tacc=[]
    for i,(op,arch) in enumerate(zip(OPS,ARCHS)):
        m,a,acc=train_model(op,arch,pool_seed*1000+i*37+11);cand.append(m);cabi.append(a);cacc.append(acc)
        tarch=ARCHS[(i+1)%len(ARCHS)];m2,a2,acc2=train_model(op,tarch,pool_seed*1000+i*37+500);targ.append(m2);tabi.append(a2);tacc.append((tarch,acc2))
    qs=make_queries(pool_seed+700,192);reg=register(cand,cabi,qs,temp,repeats,pool_seed+900)
    stochastic_c=[stochastic_token_accuracy(cand[i],cabi[i],OPS[i],temp,400,pool_seed+i) for i in range(6)]
    stochastic_t=[stochastic_token_accuracy(targ[i],tabi[i],OPS[i],temp,400,pool_seed+100+i) for i in range(6)]
    right=wrong=unres=qsum=0
    for oi in range(6):
      for tr in range(trials):
        top,p,nq=identify(targ[oi],tabi[oi],reg,qs,temp,pool_seed*100000+oi*1000+tr)
        if p<.995:unres+=1
        elif top==oi:right+=1
        else:wrong+=1
        qsum+=nq
    return {'pool_seed':pool_seed,'temperature':temp,'registration_repeats':repeats,'candidate_architectures':ARCHS,'target_architectures':[x[0] for x in tacc],'candidate_greedy_exact':cacc,'target_greedy_exact':[x[1] for x in tacc],'candidate_stochastic_semantic_acc':stochastic_c,'target_stochastic_semantic_acc':stochastic_t,'correct':right,'wrong':wrong,'unresolved':unres,'total':6*trials,'mean_runtime_relation_queries':qsum/(6*trials)}

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--pool-seed',type=int,default=811);ap.add_argument('--temperature',type=float,default=1.0);ap.add_argument('--registration-repeats',type=int,default=12);ap.add_argument('--trials',type=int,default=30);a=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);st=time.time();row=run(a.pool_seed,a.temperature,a.registration_repeats,a.trials);out={'experiment':'actual_stochastic_neural_probabilistic_registry_v1','protocol':'independently trained candidate/target neural instances; different private factorized inputs/private output permutations and cyclically changed target architectures; actual multinomial token sampling; registry stores P(raw_output(x1)==raw_output(x2)); routing uses equality bits only','row':row,'elapsed_s':time.time()-st};Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__': main()
