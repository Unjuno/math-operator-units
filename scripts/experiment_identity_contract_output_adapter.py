from __future__ import annotations
import json,random,time,math
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0,str(Path(__file__).parent))
import heldout_core as h
N=h.N;OPS=h.OPS;ARCHS=h.ARCHS
@dataclass
class FCodec:
    out:h.PrivateCodec
    pa:list[int]
    pb:list[int]
    @property
    def base(self):return self.out.base
    @property
    def reverse(self):return self.out.reverse
    @property
    def digit_to_symbol(self):return self.out.digit_to_symbol
    def encode_value(self,v):return self.out.encode_value(v)
def make_codec(rng):
    out=h.make_codec(rng);pa=list(range(N));pb=list(range(N));rng.shuffle(pa);rng.shuffle(pb);return FCodec(out,pa,pb)
class FGRU(nn.Module):
    def __init__(self,c,emb=48,hidden=96):
        super().__init__();self.codec=c;self.base=c.base;self.start=self.base;self.eos=self.base+1;self.out_vocab=self.base+2
        self.aemb=nn.Embedding(N,hidden);self.bemb=nn.Embedding(N,hidden);self.prev=nn.Embedding(self.out_vocab,emb);self.gru=nn.GRUCell(emb,hidden);self.head=nn.Linear(hidden,self.out_vocab)
    def ctx(self,p):return torch.tanh(self.aemb(p[:,0])+self.bemb(p[:,1]))
    def teacher_logits(self,p,target):
        hh=self.ctx(p);prev=torch.full((p.size(0),),self.start,dtype=torch.long,device=p.device);out=[]
        for t in range(target.size(1)):
            hh=self.gru(self.prev(prev),hh);out.append(self.head(hh));nxt=target[:,t];prev=torch.where(nxt>=0,nxt,torch.full_like(nxt,self.eos))
        return torch.stack(out,1)
    @torch.no_grad()
    def generate(self,p,max_len=h.MAX_STEPS):
        hh=self.ctx(p);prev=torch.full((p.size(0),),self.start,dtype=torch.long,device=p.device);done=torch.zeros(p.size(0),dtype=torch.bool,device=p.device);seq=[[] for _ in range(p.size(0))]
        for _ in range(max_len):
            hh=self.gru(self.prev(prev),hh);tok=self.head(hh).argmax(-1)
            for i,x in enumerate(tok.tolist()):
                if not done[i]:
                    if x==self.eos:done[i]=True
                    else:seq[i].append(x)
            prev=tok
        return seq
class FNAR(nn.Module):
    def __init__(self,c,hidden=128):
        super().__init__();self.codec=c;self.base=c.base;self.start=self.base;self.eos=self.base+1;self.out_vocab=self.base+2
        self.aemb=nn.Embedding(N,hidden);self.bemb=nn.Embedding(N,hidden);self.net=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU());self.head=nn.ModuleList([nn.Linear(hidden,self.out_vocab) for _ in range(h.MAX_STEPS)])
    def hh(self,p):return self.net(self.aemb(p[:,0])+self.bemb(p[:,1]))
    def teacher_logits(self,p,target):
        x=self.hh(p);return torch.stack([self.head[t](x) for t in range(target.size(1))],1)
    @torch.no_grad()
    def generate(self,p,max_len=h.MAX_STEPS):
        x=self.hh(p);rows=torch.stack([self.head[t](x) for t in range(min(max_len,h.MAX_STEPS))],1).argmax(-1).tolist();out=[]
        for row in rows:
            s=[]
            for tok in row:
                if tok==self.eos:break
                s.append(tok)
            out.append(s)
        return out
class FTransformer(nn.Module):
    def __init__(self,c,d=96):
        super().__init__();self.codec=c;self.base=c.base;self.start=self.base;self.eos=self.base+1;self.out_vocab=self.base+2
        self.aemb=nn.Embedding(N,d);self.bemb=nn.Embedding(N,d);self.query=nn.Parameter(torch.randn(h.MAX_STEPS,d)*.02);self.pos=nn.Parameter(torch.randn(h.MAX_STEPS+1,d)*.02)
        lay=nn.TransformerEncoderLayer(d,4,192,dropout=0,batch_first=True,activation='gelu');self.enc=nn.TransformerEncoder(lay,2);self.head=nn.ModuleList([nn.Linear(d,self.out_vocab) for _ in range(h.MAX_STEPS)])
    def teacher_logits(self,p,target):
        B=p.size(0);T=target.size(1);ctx=(self.aemb(p[:,0])+self.bemb(p[:,1])).unsqueeze(1);q=self.query[:T].unsqueeze(0).expand(B,-1,-1);x=self.enc(torch.cat([ctx,q],1)+self.pos[:T+1].unsqueeze(0))[:,1:];return torch.stack([self.head[t](x[:,t]) for t in range(T)],1)
    @torch.no_grad()
    def generate(self,p,max_len=h.MAX_STEPS):
        T=min(max_len,h.MAX_STEPS);dummy=torch.zeros((p.size(0),T),dtype=torch.long,device=p.device);rows=self.teacher_logits(p,dummy).argmax(-1).tolist();out=[]
        for row in rows:
            s=[]
            for tok in row:
                if tok==self.eos:break
                s.append(tok)
            out.append(s)
        return out
def build(arch,c):return FGRU(c) if arch=='gru' else FNAR(c) if arch=='nar_mlp' else FTransformer(c)
def rows(op,c,device):
    raw=[];ml=0
    for a in range(N):
        for b in range(N):
            y=h.op_apply(op,a,b);s=c.encode_value(y)+[c.base+1];raw.append((a,b,y,s));ml=max(ml,len(s))
    P=torch.tensor([[c.pa[a],c.pb[b]] for a,b,_,_ in raw],device=device);T=torch.full((len(raw),ml),-100,dtype=torch.long,device=device)
    for i,r in enumerate(raw):T[i,:len(r[3])]=torch.tensor(r[3],device=device)
    return raw,P,T
def train(op,arch,seed,device,steps=240):
    if arch=='transformer': steps=max(steps,600)
    rng=random.Random(seed);c=make_codec(rng);torch.manual_seed(seed);m=build(arch,c).to(device);raw,P,T=rows(op,c,device);opt=torch.optim.AdamW(m.parameters(),lr=2e-3 if arch=='transformer' else 5e-3,weight_decay=1e-5);g=torch.Generator(device=device).manual_seed(seed+77)
    for _ in range(steps):
        ix=torch.randint(0,len(raw),(256,),generator=g,device=device);log=m.teacher_logits(P[ix],T[ix]);loss=F.cross_entropy(log.reshape(-1,m.out_vocab),T[ix].reshape(-1),ignore_index=-100);opt.zero_grad();loss.backward();opt.step()
    return m,raw
def input_adapter(c):
    A={s:c.pa[s] for s in range(N)};B={s:c.pb[s] for s in range(N)};return A,B
@torch.no_grad()
def gen(m,A,B,a,b,device):return m.generate(torch.tensor([[A[a],B[b]]],device=device))[0]
def identity_state(op):
    return {'add':0,'sub':0,'min':N-1,'max':0,'xor':0,'mul':1}[op]
@torch.no_grad()
def fit_out_identity(m,op,A,B,device):
    e=identity_state(op); state_to_seq={}; seq_to_state={}
    for st in range(N):
        seq=tuple(gen(m,A,B,st,e,device));state_to_seq[st]=seq
        if seq in seq_to_state and seq_to_state[seq]!=st: raise RuntimeError('nonunique identity outputs')
        seq_to_state[seq]=st
    return {'state_to_seq':state_to_seq,'seq_to_state':seq_to_state,'identity':e}
@torch.no_grad()
def run_source(m,oad,A,B,a,b,device):
    return oad['seq_to_state'].get(tuple(gen(m,A,B,a,b,device)))
@torch.no_grad()
def action_prob(m,oad,A,B,a,b,device):
    seqs=[list(oad['state_to_seq'][y])+[m.eos] for y in range(N)];ml=max(map(len,seqs));P=torch.tensor([[A[a],B[b]]]*N,device=device);T=torch.full((N,ml),-100,dtype=torch.long,device=device)
    for i,ss in enumerate(seqs):T[i,:len(ss)]=torch.tensor(ss,device=device)
    lp=F.log_softmax(m.teacher_logits(P,T),-1);vals=[]
    for i in range(N):
        mask=T[i]>=0;pos=torch.arange(ml,device=device)[mask];vals.append(lp[i,pos,T[i][mask]].sum())
    out=torch.stack(vals).cpu();return (out-torch.logsumexp(out,0)).exp()
def pool(seed,device):
    ms=[];ads=[];ins=[];stats=[]
    for i,(op,arch) in enumerate(zip(OPS,ARCHS)):
        m,r=train(op,arch,seed*100+i*23+7,device);A,B=input_adapter(m.codec);ad=fit_out_identity(m,op,A,B,device);ms.append(m);ads.append(ad);ins.append((A,B))
        exact=sum(run_source(m,ad,A,B,a,b,device)==h.op_apply(op,a,b) for a in range(N) for b in range(N))/(N*N);stats.append((op,arch,exact))
    score={}
    for op in OPS:
        exp=[h.op_apply(op,a,b) for a,b in h.PROBES];score[op]=[]
        for m,ad,(A,B) in zip(ms,ads,ins):score[op].append(sum(run_source(m,ad,A,B,a,b,device)==y for (a,b),y in zip(h.PROBES,exp))/len(exp))
    pos=[score[op][i] for i,op in enumerate(h.SEEN_OPS)];neg=[score[op][j] for i,op in enumerate(h.SEEN_OPS) for j in range(len(h.SEEN_OPS)) if j!=i];th=(min(pos)+max(neg))/2;routes={}
    for op in OPS:
        o=sorted(range(6),key=lambda i:score[op][i],reverse=True);routes[op]=(o[0],o[1])
    cache={}
    def prob(i,a,b):
        k=(i,a,b)
        if k not in cache:
            A,B=ins[i];cache[k]=action_prob(ms[i],ads[i],A,B,a,b,device)
        return cache[k]
    rng=random.Random(seed+8000);res={}
    for d in range(1,6):
        hard=soft=0
        for _ in range(100):
            vh=vs=truth=rng.randrange(N)
            for __ in range(d):
                op=rng.choice(OPS);b=rng.randrange(N);p,s=routes[op];A,B=ins[p];vh=run_source(ms[p],ads[p],A,B,vh,b,device);vs=int((.6*prob(p,vs,b)+.4*prob(s,vs,b)).argmax());truth=h.op_apply(op,truth,b)
            hard+=vh==truth;soft+=vs==truth
        res[str(d)]={'hard':hard/100,'soft06':soft/100}
    return {'seed':seed,'source_exact':stats,'threshold':th,'heldout_route':{'top':routes['mul'][0],'second':routes['mul'][1],'scores':score['mul']},'depth':res}
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,nargs='+',default=[710,711,712]);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);dev=torch.device('cpu');st=time.time();ps=[]
    for seed in a.seeds:
        print('POOL',seed,flush=True);r=pool(seed,dev);ps.append(r);print(json.dumps(r,indent=2),flush=True)
    out={'experiment':'identity_contract_output_adapter_v1','input_adapter':'32 diagonal native-input examples recover both operand-state permutations; no output labels used','output_adapter':'32 identity-contract probes source(s,e)=s map private output sequences to common input handles; no exact output/action labels are supplied to adapter','pools':ps,'elapsed_s':time.time()-st};p=Path(a.out);p.write_text(json.dumps(out,indent=2));print('WROTE',p)
if __name__=='__main__':main()
