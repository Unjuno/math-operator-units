from __future__ import annotations
import argparse, importlib.util, json, random, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE=Path(__file__).with_name('experiment_joint_opaque_positionfree_heldout_transformer.py')
spec=importlib.util.spec_from_file_location('jointbase',BASE)
j=importlib.util.module_from_spec(spec); sys.modules['jointbase']=j; spec.loader.exec_module(j); h=j.h

STAGE_VARIANTS=7
LITERALS=['TASK','ARG','DO','[',']','(',')','ON','NOW','USE']
C1='<C1>'; C2='<C2>'; VAL='<VAL>'


def render_stage(code,b,v):
    c1,c2=code; x=str(b)
    if v==0: return [c1,c2,'ARG',x]
    if v==1: return ['TASK',c1,c2,x]
    if v==2: return [x,'DO',c1,c2]
    if v==3: return ['[',c1,c2,']',x]
    if v==4: return [x,'(',c1,c2,')']
    if v==5: return [c1,c2,'ON',x,'NOW']
    if v==6: return ['USE',x,c1,c2]
    raise KeyError(v)


def normalize_stream(tokens,grounded_codes):
    out=[]; i=0
    while i<len(tokens):
        if i+1<len(tokens) and (tokens[i],tokens[i+1]) in grounded_codes:
            out.extend([C1,C2]); i+=2
        elif tokens[i].lstrip('-').isdigit(): out.append(VAL); i+=1
        else: out.append(tokens[i]); i+=1
    return out

VOC=[C1,C2,VAL]+LITERALS
TOK={t:i for i,t in enumerate(VOC)}

class Segmenter(nn.Module):
    def __init__(self):
        super().__init__(); self.emb=nn.Embedding(len(VOC),24); self.gru=nn.GRU(24,64,batch_first=True); self.head=nn.Linear(64,1)
    def forward(self,x,lengths=None):
        y,_=self.gru(self.emb(x)); return self.head(y).squeeze(-1)


def gen_stream(rng,codes,ops,depth):
    toks=[]; ends=[]; truth=[]
    for _ in range(depth):
        op=rng.choice(ops); b=rng.randrange(h.N); v=rng.randrange(STAGE_VARIANTS); chunk=render_stage(codes[op],b,v); toks.extend(chunk); ends.append(len(toks)-1); truth.append((op,b,v))
    return toks,ends,truth


def train_segmenter(seed=880000,steps=1200,batch=96):
    rng=random.Random(seed); torch.manual_seed(seed); model=Segmenter(); opt=torch.optim.AdamW(model.parameters(),lr=3e-3); dummy={op:(C1,C2) for op in h.SEEN_OPS}
    for _ in range(steps):
        rows=[]
        for _ in range(batch):
            depth=rng.randint(1,3); toks,ends,_=gen_stream(rng,dummy,h.SEEN_OPS,depth); ids=[TOK[t if not t.lstrip('-').isdigit() else VAL] for t in toks]; lab=[0.0]*len(ids)
            for e in ends: lab[e]=1.0
            rows.append((ids,lab))
        mx=max(len(r[0]) for r in rows); x=torch.zeros((batch,mx),dtype=torch.long); y=torch.zeros((batch,mx)); mask=torch.zeros((batch,mx),dtype=torch.bool)
        for i,(ids,lab) in enumerate(rows): x[i,:len(ids)]=torch.tensor(ids); y[i,:len(ids)]=torch.tensor(lab); mask[i,:len(ids)]=True
        logits=model(x); loss=F.binary_cross_entropy_with_logits(logits[mask],y[mask],pos_weight=torch.tensor(4.0)); opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def predict_ends(model,norm):
    x=torch.tensor([[TOK[t] for t in norm]],dtype=torch.long); p=torch.sigmoid(model(x)[0]); return [i for i,v in enumerate(p.tolist()) if v>=0.5],p.tolist()


def parser_eval(model,codes,seed=881000,n=300):
    rng=random.Random(seed); grounded=set(codes.values()); out={}
    for depth in range(1,11):
        exact=tp=predn=goldn=0
        for _ in range(n):
            toks,ends,_=gen_stream(rng,codes,h.OPS,depth); pred,_=predict_ends(model,normalize_stream(toks,grounded)); exact+=pred==ends; tp+=len(set(pred)&set(ends)); predn+=len(pred); goldn+=len(ends)
        out[str(depth)]={'exact_segmentation':exact/n,'boundary_precision':tp/max(predn,1),'boundary_recall':tp/goldn}
    return out


def decode_segments(tokens,ends,grounded_codes,code_to_route):
    chunks=[]; start=0
    for end in ends:
        if end<start or end>=len(tokens): return None
        chunk=tokens[start:end+1]; start=end+1; found=[]
        for i in range(len(chunk)-1):
            pair=(chunk[i],chunk[i+1])
            if pair in grounded_codes: found.append(pair)
        nums=[int(t) for t in chunk if t.lstrip('-').isdigit()]
        if len(found)!=1 or len(nums)!=1 or found[0] not in code_to_route: return None
        chunks.append((found[0],nums[0]))
    if start!=len(tokens): return None
    return chunks


def e2e_seed(model,seed,device,steps=220,n=100):
    rng=random.Random(seed+5555); codes=j.make_codes(rng); models,_,ads,_,_=j.train_pool(seed,device,steps); _,threshold,routes=j.behavioral_code_grounding(codes,models,ads,device); grounded=set(codes.values()); out={}; erng=random.Random(seed+99000)
    for depth in range(1,11):
        segok=e2e=routeok=stages=0
        for _ in range(n):
            value=truth=erng.randrange(h.N); toks,gold,plan=gen_stream(erng,codes,h.OPS,depth); pred,_=predict_ends(model,normalize_stream(toks,grounded)); segok+=pred==gold; chunks=decode_segments(toks,pred,grounded,routes)
            if chunks is None: continue
            for (code,b),(op,_,_) in zip(chunks,plan):
                rr=routes[code]; routeok+=rr['top']==h.OPS.index(op); stages+=1; value=h.run_source(models[rr['top']],ads[rr['top']],value,b,device); truth=h.op_apply(op,truth,b)
            e2e+=len(chunks)==depth and value==truth
        out[str(depth)]={'segmentation_exact':segok/n,'e2e':e2e/n,'route_acc_given_segments':routeok/max(stages,1)}
    return {'seed':seed,'threshold':threshold,'result':out}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); ap.add_argument('--steps',type=int,default=220); a=ap.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1); start=time.time(); segmenter=train_segmenter(); parser=parser_eval(segmenter,j.make_codes(random.Random(123))); device=torch.device('cpu'); pools=[e2e_seed(segmenter,seed,device,a.steps,100) for seed in a.seeds]; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps({'experiment':'learned_stream_stage_segmentation','segmenter_training_depths':[1,2,3],'parser':parser,'pools':pools,'elapsed_s':time.time()-start},indent=2))

if __name__=='__main__': main()
