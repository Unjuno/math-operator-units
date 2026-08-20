from __future__ import annotations
import argparse, json, random, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

N=64
OPS=("add","sub","min","max","xor")

def op_apply(op,a,b):
    if op=="add": return (a+b)%N
    if op=="sub": return (a-b)%N
    if op=="min": return min(a,b)
    if op=="max": return max(a,b)
    if op=="xor": return a^b
    raise KeyError(op)

def digits_base(v,base):
    if v==0:return [0]
    ds=[]
    while v: ds.append(v%base); v//=base
    return list(reversed(ds))

@dataclass
class PrivateCodec:
    base:int; reverse:bool; digit_to_symbol:List[int]; pair_perm:List[int]
    def encode_value(self,v):
        ds=digits_base(v,self.base)
        if self.reverse: ds=list(reversed(ds))
        return [self.digit_to_symbol[d] for d in ds]
    def pair_id(self,a,b): return self.pair_perm[a*N+b]

def make_codec(rng):
    base=rng.randint(5,12); sy=list(range(base)); rng.shuffle(sy); pp=list(range(N*N)); rng.shuffle(pp)
    return PrivateCodec(base,bool(rng.getrandbits(1)),sy,pp)

class Specialist(nn.Module):
    def __init__(self,codec,emb=64,hidden=128):
        super().__init__(); self.codec=codec; self.base=codec.base; self.start=self.base; self.eos=self.base+1; self.out_vocab=self.base+2
        self.pair_emb=nn.Embedding(N*N,hidden)
        self.prev_emb=nn.Embedding(self.out_vocab,emb)
        self.gru=nn.GRUCell(emb,hidden); self.head=nn.Linear(hidden,self.out_vocab)
    def init_h(self,pair): return torch.tanh(self.pair_emb(pair))
    def teacher_logits(self,pair,target):
        h=self.init_h(pair); prev=torch.full_like(pair,self.start); outs=[]
        for t in range(target.size(1)):
            h=self.gru(self.prev_emb(prev),h); outs.append(self.head(h)); nxt=target[:,t]
            prev=torch.where(nxt>=0,nxt,torch.full_like(nxt,self.eos))
        return torch.stack(outs,1)
    @torch.no_grad()
    def generate(self,pair,max_len=5):
        h=self.init_h(pair); prev=torch.full_like(pair,self.start); done=torch.zeros_like(pair,dtype=torch.bool); seqs=[[] for _ in range(pair.numel())]
        for _ in range(max_len):
            h=self.gru(self.prev_emb(prev),h); tok=self.head(h).argmax(-1)
            for i,x in enumerate(tok.tolist()):
                if not done[i]:
                    if x==self.eos: done[i]=True
                    else: seqs[i].append(x)
            prev=tok
            if bool(done.all()): break
        return seqs

def make_rows(op,codec,device):
    raw=[]; maxlen=0
    for a in range(N):
        for b in range(N):
            y=op_apply(op,a,b); seq=codec.encode_value(y)+[codec.base+1]; maxlen=max(maxlen,len(seq)); raw.append((codec.pair_id(a,b),a,b,y,seq))
    P=torch.tensor([r[0] for r in raw],device=device); T=torch.full((len(raw),maxlen),-100,dtype=torch.long,device=device)
    for i,r in enumerate(raw):T[i,:len(r[4])]=torch.tensor(r[4],device=device)
    return raw,P,T

def train_one(op,seed,device,steps=300,batch=512):
    rng=random.Random(seed); codec=make_codec(rng); torch.manual_seed(seed); m=Specialist(codec).to(device); rows,P,T=make_rows(op,codec,device)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3,weight_decay=1e-5); g=torch.Generator(device=device); g.manual_seed(seed+77); losses=[]
    for st in range(steps):
        idx=torch.randint(0,len(rows),(batch,),generator=g,device=device); logits=m.teacher_logits(P[idx],T[idx]); loss=F.cross_entropy(logits.reshape(-1,m.out_vocab),T[idx].reshape(-1),ignore_index=-100)
        opt.zero_grad();loss.backward();opt.step()
        if st in (0,steps//2,steps-1):losses.append(float(loss.detach()))
    return m,rows,losses

@torch.no_grad()
def gen_rows(m,rows,idx,device):
    out=[]
    for st in range(0,len(idx),1024):
        ch=idx[st:st+1024]; p=torch.tensor([rows[i][0] for i in ch],device=device); out.extend(m.generate(p))
    return out

@torch.no_grad()
def exact_acc(m,rows,device):
    idx=list(range(len(rows))); ss=gen_rows(m,rows,idx,device); return sum(s==m.codec.encode_value(rows[i][3]) for i,s in zip(idx,ss))/len(idx)

def infer_adapter(anchors,base_min=5,base_max=12):
    best=None; symbols=sorted(set(s for _,seq in anchors for s in seq)); smap={s:i for i,s in enumerate(symbols)}
    for base in range(base_min,base_max+1):
      for rev in (False,True):
        pairs=[]; lenok=0; pos=0; lp=0
        for y,seq in anchors:
            ds=digits_base(y,base); ds=list(reversed(ds)) if rev else ds
            if len(ds)!=len(seq): lp+=abs(len(ds)-len(seq))+1; continue
            lenok+=1; pos+=len(ds); pairs += list(zip(ds,seq))
        cols=max(base,len(symbols)); cnt=[[0]*cols for _ in range(base)]
        for d,s in pairs: cnt[d][smap[s]]+=1
        dp={0:(0,[])}
        for d in range(base):
            nd={}
            for mask,(sc,ass) in dp.items():
                for c in range(cols):
                    if mask>>c&1:continue
                    k=mask|1<<c; v=sc+cnt[d][c]
                    if k not in nd or v>nd[k][0]:nd[k]=(v,ass+[c])
            dp=nd
        match,ass=max(dp.values(),key=lambda z:z[0]); score=match-3*lp; cand=(score,match,lenok,pos,-lp,base,rev,ass,symbols)
        if best is None or cand[:5]>best[:5]:best=cand
    _,match,lenok,pos,nlp,base,rev,ass,symbols=best; col={i:s for i,s in enumerate(symbols)}; std={col[c]:d for d,c in enumerate(ass) if c<len(symbols)}
    return {"base":base,"reverse":rev,"sym_to_digit":std,"fit_match":match,"fit_positions":pos,"fit_len_ok":lenok,"fit_length_penalty":-nlp}

def decode(ad,seq):
    try:ds=[ad['sym_to_digit'][s] for s in seq]
    except KeyError:return None
    if ad['reverse']:ds=list(reversed(ds))
    v=0
    for d in ds:
        if d>=ad['base']:return None
        v=v*ad['base']+d
    return v

def active_anchor_indices(rows):
    vals=list(range(12))+[13,17,23,29,37,47,59,63]
    first={}
    for i,r in enumerate(rows): first.setdefault(r[3],i)
    return [first[v] for v in vals if v in first]

def adapter_trial(m,rows,device,mode,n,seed,testn=1200):
    rng=random.Random(seed)
    if mode=='active': idx=active_anchor_indices(rows)
    else: idx=rng.sample(range(len(rows)),n)
    seqs=gen_rows(m,rows,idx,device); ad=infer_adapter([(rows[i][3],s) for i,s in zip(idx,seqs)])
    used=set(idx); pool=[i for i in range(len(rows)) if i not in used]; ti=rng.sample(pool,min(testn,len(pool))); ts=gen_rows(m,rows,ti,device)
    acc=sum(decode(ad,s)==rows[i][3] for i,s in zip(ti,ts))/len(ti)
    c=m.codec; struct=ad['base']==c.base and ad['reverse']==c.reverse and all(ad['sym_to_digit'].get(c.digit_to_symbol[d])==d for d in range(c.base))
    return acc,struct,ad,len(idx)

@torch.no_grad()
def run_source(m,ad,a,b,device):
    p=torch.tensor([m.codec.pair_id(a,b)],device=device); return decode(ad,m.generate(p)[0])

def eval_programs(models,ads,device,nprog=300,seed=0):
    rng=random.Random(seed); out={}
    for depth in range(1,6):
        e2e=0; local=0; stages=0
        for _ in range(nprog):
            truth=pred=rng.randrange(N); prog=[(rng.choice(OPS),rng.randrange(N)) for __ in range(depth)]
            for op,b in prog:
                i=OPS.index(op); expected=op_apply(op,pred,b); y=run_source(models[i],ads[i],pred,b,device); stages+=1;local+=int(y==expected)
                pred=-1 if y is None else y; truth=op_apply(op,truth,b)
            e2e+=int(pred==truth)
        out[str(depth)]={"e2e":e2e/nprog,"stage_local":local/stages}
    return out

def pool(seed,device,steps,reps):
    models=[]; rowsall=[]; tr=[]
    for i,op in enumerate(OPS):
        m,rows,loss=train_one(op,seed*100+i*19+3,device,steps); acc=exact_acc(m,rows,device); models.append(m);rowsall.append(rows);tr.append({"op":op,"codec":asdict(m.codec),"loss":loss,"exact":acc})
    stats={}
    for mode,n in [('random8',8),('random16',16),('random32',32),('active20',20)]:
        vals=[];ss=[];asz=[]
        for i,(m,rows) in enumerate(zip(models,rowsall)):
            rr=1 if mode=='active20' else reps
            for r in range(rr):
                a,s,_,an=adapter_trial(m,rows,device,'active' if mode=='active20' else 'random',n,seed*10000+i*100+r);vals.append(a);ss.append(s);asz.append(an)
        stats[mode]={"mean_decode_exact":sum(vals)/len(vals),"min_decode_exact":min(vals),"structural_rate":sum(ss)/len(ss),"trials":len(vals),"anchor_count_mean":sum(asz)/len(asz)}
    ads=[]
    for m,rows in zip(models,rowsall):
        idx=active_anchor_indices(rows); seq=gen_rows(m,rows,idx,device); ads.append(infer_adapter([(rows[i][3],s) for i,s in zip(idx,seq)]))
    progs=eval_programs(models,ads,device,300,seed+9999)
    return {"pool_seed":seed,"train":tr,"adapter_stats":stats,"programs":progs}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--pools',type=int,default=3);ap.add_argument('--steps',type=int,default=320);ap.add_argument('--reps',type=int,default=6);a=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device('cpu');rs=[];t=time.time()
    for p in range(a.pools):
        print('POOL',p,flush=True);r=pool(120+p,device,a.steps,a.reps);rs.append(r);print(json.dumps({"train":[x['exact'] for x in r['train']],"ad":r['adapter_stats'],"prog":r['programs']},indent=2),flush=True)
    agg={k:{"mean_decode_exact":sum(r['adapter_stats'][k]['mean_decode_exact'] for r in rs)/len(rs),"worst_trial":min(r['adapter_stats'][k]['min_decode_exact'] for r in rs),"mean_structural_rate":sum(r['adapter_stats'][k]['structural_rate'] for r in rs)/len(rs)} for k in rs[0]['adapter_stats']}
    pg={d:{"mean_e2e":sum(r['programs'][d]['e2e'] for r in rs)/len(rs),"min_pool_e2e":min(r['programs'][d]['e2e'] for r in rs),"mean_stage_local":sum(r['programs'][d]['stage_local'] for r in rs)/len(rs)} for d in rs[0]['programs']}
    out={"experiment":"neural_private_tokenizer_adapter_v2","claim_scope":"output/input ABI isolation screening; private pair-tokenizer per neural source; private radix/digit/reversal output vocab and output-head size; fusion receives behavior-output anchors only","ops":list(OPS),"domain":N,"pools":rs,"aggregate":{"adapter":agg,"programs":pg},"elapsed_s":time.time()-t}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2));print('WROTE',a.out)
if __name__=='__main__':main()
