from __future__ import annotations
import argparse, json, math, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

N=32
SEEN_OPS=("add","sub","min","max","xor")
HELDOUT_OP="mul"
OPS=SEEN_OPS+(HELDOUT_OP,)
ARCHS=("gru","nar_mlp","gru","nar_mlp","gru","transformer")
MAX_STEPS=4

def op_apply(op,a,b):
    if op=="add": return (a+b)%N
    if op=="sub": return (a-b)%N
    if op=="min": return min(a,b)
    if op=="max": return max(a,b)
    if op=="xor": return a^b
    if op=="mul": return (a*b)%N
    raise KeyError(op)

def digits_base(v,base):
    if v==0: return [0]
    ds=[]
    while v:
        ds.append(v%base); v//=base
    return list(reversed(ds))

@dataclass
class PrivateCodec:
    base:int
    reverse:bool
    digit_to_symbol:List[int]
    pair_perm:List[int]
    def encode_value(self,v):
        ds=digits_base(v,self.base)
        if self.reverse: ds=list(reversed(ds))
        return [self.digit_to_symbol[d] for d in ds]
    def pair_id(self,a,b): return self.pair_perm[a*N+b]

def make_codec(rng):
    base=rng.randint(5,10); sy=list(range(base)); rng.shuffle(sy); pp=list(range(N*N)); rng.shuffle(pp)
    return PrivateCodec(base,bool(rng.getrandbits(1)),sy,pp)

class GRUSpecialist(nn.Module):
    def __init__(self,codec,emb=48,hidden=96):
        super().__init__(); self.codec=codec; self.base=codec.base; self.start=self.base; self.eos=self.base+1; self.out_vocab=self.base+2
        self.pair_emb=nn.Embedding(N*N,hidden); self.prev_emb=nn.Embedding(self.out_vocab,emb)
        self.gru=nn.GRUCell(emb,hidden); self.head=nn.Linear(hidden,self.out_vocab)
    def teacher_logits(self,pair,target):
        h=torch.tanh(self.pair_emb(pair)); prev=torch.full_like(pair,self.start); outs=[]
        for t in range(target.size(1)):
            h=self.gru(self.prev_emb(prev),h); outs.append(self.head(h)); nxt=target[:,t]
            prev=torch.where(nxt>=0,nxt,torch.full_like(nxt,self.eos))
        return torch.stack(outs,1)
    @torch.no_grad()
    def generate(self,pair,max_len=MAX_STEPS):
        h=torch.tanh(self.pair_emb(pair)); prev=torch.full_like(pair,self.start); done=torch.zeros_like(pair,dtype=torch.bool); seqs=[[] for _ in range(pair.numel())]
        for _ in range(max_len):
            h=self.gru(self.prev_emb(prev),h); tok=self.head(h).argmax(-1)
            for i,x in enumerate(tok.tolist()):
                if not done[i]:
                    if x==self.eos: done[i]=True
                    else: seqs[i].append(x)
            prev=tok
        return seqs

class NARMLP(nn.Module):
    def __init__(self,codec,hidden=128):
        super().__init__(); self.codec=codec; self.base=codec.base; self.start=self.base; self.eos=self.base+1; self.out_vocab=self.base+2
        self.pair_emb=nn.Embedding(N*N,hidden)
        self.net=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.head=nn.ModuleList([nn.Linear(hidden,self.out_vocab) for _ in range(MAX_STEPS)])
    def teacher_logits(self,pair,target):
        h=self.net(self.pair_emb(pair)); return torch.stack([self.head[t](h) for t in range(target.size(1))],1)
    @torch.no_grad()
    def generate(self,pair,max_len=MAX_STEPS):
        h=self.net(self.pair_emb(pair)); logits=torch.stack([self.head[t](h) for t in range(min(MAX_STEPS,max_len))],1)
        rows=logits.argmax(-1).tolist(); out=[]
        for row in rows:
            s=[]
            for tok in row:
                if tok==self.eos: break
                s.append(tok)
            out.append(s)
        return out

class TransformerSpecialist(nn.Module):
    """Attention-based held-out specialist with a private pair vocabulary and output ABI."""
    def __init__(self,codec,d_model=96,nhead=4,layers=2):
        super().__init__(); self.codec=codec; self.base=codec.base; self.start=self.base; self.eos=self.base+1; self.out_vocab=self.base+2
        self.pair_emb=nn.Embedding(N*N,d_model)
        self.query=nn.Parameter(torch.randn(MAX_STEPS,d_model)*0.02)
        self.pos=nn.Parameter(torch.randn(MAX_STEPS+1,d_model)*0.02)
        layer=nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward=192,dropout=0.0,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,layers)
        self.head=nn.ModuleList([nn.Linear(d_model,self.out_vocab) for _ in range(MAX_STEPS)])
    def teacher_logits(self,pair,target):
        B=pair.shape[0]; T=target.size(1); pairtok=self.pair_emb(pair).unsqueeze(1); q=self.query[:T].unsqueeze(0).expand(B,-1,-1)
        h=self.encoder(torch.cat([pairtok,q],1)+self.pos[:T+1].unsqueeze(0))[:,1:,:]
        return torch.stack([self.head[t](h[:,t]) for t in range(T)],1)
    @torch.no_grad()
    def generate(self,pair,max_len=MAX_STEPS):
        T=min(MAX_STEPS,max_len); dummy=torch.zeros((pair.shape[0],T),dtype=torch.long,device=pair.device)
        rows=self.teacher_logits(pair,dummy).argmax(-1).tolist(); out=[]
        for row in rows:
            s=[]
            for tok in row:
                if tok==self.eos: break
                s.append(tok)
            out.append(s)
        return out

def make_rows(op,codec,device):
    raw=[]; maxlen=0
    for a in range(N):
        for b in range(N):
            y=op_apply(op,a,b); seq=codec.encode_value(y)+[codec.base+1]; maxlen=max(maxlen,len(seq)); raw.append((codec.pair_id(a,b),a,b,y,seq))
    P=torch.tensor([r[0] for r in raw],device=device); T=torch.full((len(raw),maxlen),-100,dtype=torch.long,device=device)
    for i,r in enumerate(raw): T[i,:len(r[4])]=torch.tensor(r[4],device=device)
    return raw,P,T

def build_model(arch,codec):
    if arch=='gru': return GRUSpecialist(codec)
    if arch=='nar_mlp': return NARMLP(codec)
    if arch=='transformer': return TransformerSpecialist(codec)
    raise KeyError(arch)

def train_one(op,arch,seed,device,steps=300,batch=256):
    rng=random.Random(seed); codec=make_codec(rng); torch.manual_seed(seed); model=build_model(arch,codec).to(device); rows,P,T=make_rows(op,codec,device)
    opt=torch.optim.AdamW(model.parameters(),lr=4e-3 if arch=='transformer' else 5e-3,weight_decay=1e-5); g=torch.Generator(device=device); g.manual_seed(seed+77); loss_hist=[]
    for st in range(steps):
        idx=torch.randint(0,len(rows),(batch,),generator=g,device=device); logits=model.teacher_logits(P[idx],T[idx])
        loss=F.cross_entropy(logits.reshape(-1,model.out_vocab),T[idx].reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
        if st in (0,steps//2,steps-1): loss_hist.append(float(loss.detach()))
    return model,rows,loss_hist

@torch.no_grad()
def gen_rows(m,rows,idx,device):
    out=[]
    for st in range(0,len(idx),512):
        ch=idx[st:st+512]; pair=torch.tensor([rows[i][0] for i in ch],device=device); out.extend(m.generate(pair))
    return out

@torch.no_grad()
def exact_acc(m,rows,device):
    idx=list(range(len(rows))); ss=gen_rows(m,rows,idx,device); return sum(s==m.codec.encode_value(rows[i][3]) for i,s in zip(idx,ss))/len(idx)

def infer_adapter(anchors,base_min=5,base_max=10):
    best=None
    for base in range(base_min,base_max+1):
        for rev in (False,True):
            d2s={}; s2d={}; matches=0; conflicts=0; lenpen=0
            for y,seq in anchors:
                ds=digits_base(y,base); ds=list(reversed(ds)) if rev else ds
                if len(ds)!=len(seq): lenpen+=abs(len(ds)-len(seq))+1; continue
                for d,s in zip(ds,seq):
                    if (d in d2s and d2s[d]!=s) or (s in s2d and s2d[s]!=d): conflicts+=1
                    else: d2s[d]=s; s2d[s]=d; matches+=1
            score=(matches-20*conflicts-5*lenpen,matches,-conflicts,-lenpen)
            if best is None or score>best[0]: best=(score,base,rev,s2d)
    _,base,rev,s2d=best; return {'base':base,'reverse':rev,'sym_to_digit':dict(s2d)}

def decode(ad,seq):
    try: ds=[ad['sym_to_digit'][s] for s in seq]
    except KeyError: return None
    if ad['reverse']: ds=list(reversed(ds))
    val=0
    for d in ds:
        if d>=ad['base']: return None
        val=val*ad['base']+d
    return val

def active_anchor_indices(rows):
    vals=list(range(12))+[13,17,19,23,29,31]; first={}
    for i,r in enumerate(rows): first.setdefault(r[3],i)
    return [first[x] for x in vals if x in first]

def fit_adapter(model,rows,device):
    idx=active_anchor_indices(rows); seqs=gen_rows(model,rows,idx,device); ad=infer_adapter([(rows[i][3],s) for i,s in zip(idx,seqs)]); codec=model.codec
    structural=(ad['base']==codec.base and ad['reverse']==codec.reverse and all(ad['sym_to_digit'].get(codec.digit_to_symbol[d])==d for d in range(codec.base)))
    all_idx=list(range(len(rows))); allseq=gen_rows(model,rows,all_idx,device); acc=sum(decode(ad,s)==rows[i][3] for i,s in zip(all_idx,allseq))/len(all_idx)
    return ad,structural,acc,len(idx)

@torch.no_grad()
def run_source(model,ad,a,b,device):
    pair=torch.tensor([model.codec.pair_id(a,b)],device=device); return decode(ad,model.generate(pair)[0])

@torch.no_grad()
def action_logprobs(model,ad,a,b,device):
    inv={digit:sym for sym,digit in ad['sym_to_digit'].items()}; valid=[]; seqs=[]; maxlen=0
    for y in range(N):
        ds=digits_base(y,ad['base']); ds=list(reversed(ds)) if ad['reverse'] else ds
        try: seq=[inv[d] for d in ds]+[model.eos]
        except KeyError: continue
        valid.append(y); seqs.append(seq); maxlen=max(maxlen,len(seq))
    pair=torch.tensor([model.codec.pair_id(a,b)]*len(valid),device=device); target=torch.full((len(valid),maxlen),-100,dtype=torch.long,device=device)
    for i,seq in enumerate(seqs): target[i,:len(seq)]=torch.tensor(seq,device=device)
    lp=F.log_softmax(model.teacher_logits(pair,target),-1); vals=[]
    for i in range(len(valid)):
        mask=target[i]>=0; pos=torch.arange(maxlen,device=device)[mask]; vals.append(lp[i,pos,target[i][mask]].sum())
    out=torch.full((N,),float('-inf'),device=device); out[torch.tensor(valid,device=device)]=torch.stack(vals); out-=torch.logsumexp(out,0); return out.cpu()

PROBES=[(0,0),(1,2),(3,5),(7,11),(13,4),(19,6),(23,9),(31,15),(16,3),(5,27)]

def command_grounding(models,ads,device):
    score={}
    for op in OPS:
        expected=[op_apply(op,a,b) for a,b in PROBES]; score[op]=[]
        for m,ad in zip(models,ads):
            pred=[run_source(m,ad,a,b,device) for a,b in PROBES]; score[op].append(sum(int(x==y) for x,y in zip(pred,expected))/len(PROBES))
    pos=[score[op][i] for i,op in enumerate(SEEN_OPS)]; neg=[score[op][j] for i,op in enumerate(SEEN_OPS) for j in range(len(SEEN_OPS)) if j!=i]
    lo=min(pos); hi=max(neg); threshold=(lo+hi)/2; routing={}
    for op in OPS:
        order=sorted(range(len(models)),key=lambda j:score[op][j],reverse=True)
        routing[op]={'top':order[0],'second':order[1],'top_score':score[op][order[0]],'second_score':score[op][order[1]],'accepted':score[op][order[0]]>=threshold}
    return score,threshold,routing,{'seen_positive_min':lo,'seen_negative_max':hi}

def fuse(lp1,lp2,alpha): return torch.logaddexp(lp1+math.log(alpha),lp2+math.log(1-alpha))

def eval_programs(models,ads,routing,device,nprog=250,seed=0,alpha=None):
    rng=random.Random(seed); out={}; cache={}
    def getlp(i,a,b):
        key=(i,a,b)
        if key not in cache: cache[key]=action_logprobs(models[i],ads[i],a,b,device)
        return cache[key]
    for depth in range(1,6):
        e2e=local=stages=involved=involved_ok=route_ok=0
        for _ in range(nprog):
            pred=truth=rng.randrange(N); prog=[(rng.choice(OPS),rng.randrange(N)) for __ in range(depth)]; has_held=any(op==HELDOUT_OP for op,_ in prog); involved+=int(has_held); ok=True
            for op,b in prog:
                rr=routing[op]; primary=rr['top']; secondary=rr['second']; route_ok+=int(primary==OPS.index(op)); stages+=1; expected_local=op_apply(op,pred,b)
                y=run_source(models[primary],ads[primary],pred,b,device) if alpha is None else int(fuse(getlp(primary,pred,b),getlp(secondary,pred,b),alpha).argmax())
                local+=int(y==expected_local); pred=-1 if y is None else y; truth=op_apply(op,truth,b)
                if pred<0: ok=False; pred=0
            good=ok and pred==truth; e2e+=int(good); involved_ok+=int(has_held and good)
        out[str(depth)]={'e2e':e2e/nprog,'stage_local':local/stages,'route_acc':route_ok/stages,'heldout_involved_e2e':involved_ok/max(involved,1),'heldout_involved_n':involved}
    return out

def run_pool(seed,device,steps):
    models=[]; rowsall=[]; train=[]
    for i,(op,arch) in enumerate(zip(OPS,ARCHS)):
        m,rows,loss=train_one(op,arch,seed*100+i*23+7,device,steps=steps); acc=exact_acc(m,rows,device); models.append(m); rowsall.append(rows)
        train.append({'op':op,'arch':arch,'exact':acc,'head':m.out_vocab,'codec':{'base':m.codec.base,'reverse':m.codec.reverse},'loss':loss})
    ads=[]; adstats=[]
    for m,rows in zip(models,rowsall):
        ad,struct,decacc,n=fit_adapter(m,rows,device); ads.append(ad); adstats.append({'structural':struct,'decode_exact':decacc,'anchors':n})
    scores,threshold,routing,cal=command_grounding(models,ads,device)
    return {'seed':seed,'train':train,'adapters':adstats,'seen_only_threshold':threshold,'threshold_calibration':cal,'routing':routing,'score_matrix':scores,'hard':eval_programs(models,ads,routing,device,250,seed+10000,None),'soft_0.6':eval_programs(models,ads,routing,device,120,seed+20000,0.6),'soft_0.5':eval_programs(models,ads,routing,device,120,seed+20000,0.5)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); ap.add_argument('--steps',type=int,default=320); args=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); started=time.time(); pools=[]
    for seed in args.seeds:
        print('POOL',seed,flush=True); r=run_pool(seed,device,args.steps); pools.append(r); print(json.dumps({'train':[(x['op'],x['arch'],x['exact']) for x in r['train']],'adapters':r['adapters'],'threshold':r['seen_only_threshold'],'routing':r['routing'],'hard':r['hard'],'soft06':r['soft_0.6']},indent=2),flush=True)
    result={'experiment':'heldout_transformer_private_abi_insertion','domain':N,'seen_ops':SEEN_OPS,'heldout_op':HELDOUT_OP,'architectures':ARCHS,'protocol':'held-out Transformer/source excluded from admission-threshold calibration; inserted using only source-specific behavioral ABI adapter plus opaque-command behavior probes; all outputs mapped into common action space before fusion','pools':pools,'elapsed_s':time.time()-started}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2)); print('WROTE',args.out,flush=True)

if __name__=='__main__': main()
