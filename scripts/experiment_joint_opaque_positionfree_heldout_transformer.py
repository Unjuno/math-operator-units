from __future__ import annotations
import argparse, importlib.util, json, math, random, time, sys
from collections import Counter, defaultdict
from pathlib import Path
import torch

BASE=Path(__file__).with_name('experiment_heldout_transformer_private_abi_insertion.py')
spec=importlib.util.spec_from_file_location('heldout_base',BASE)
h=importlib.util.module_from_spec(spec); sys.modules['heldout_base']=h; spec.loader.exec_module(h)

VARIANTS=7

def make_codes(rng):
    toks=[f'Q{i}' for i in range(40)]
    rng.shuffle(toks)
    return {op:(toks[2*i],toks[2*i+1]) for i,op in enumerate(h.OPS)}

def render(code,a,b,variant):
    c1,c2=code; a=str(a); b=str(b)
    if variant==0: return ['BOS',c1,c2,'A',a,'B',b,'RESP']
    if variant==1: return ['BOS','TASK',c1,c2,'A',a,'B',b,'RESP']
    if variant==2: return ['BOS','A',a,'B',b,'DO',c1,c2,'RESP']
    if variant==3: return ['BOS','[',c1,c2,']','A',a,'B',b,'RESP']
    if variant==4: return ['BOS','A',a,'WITH',b,'(',c1,c2,')','RESP']
    if variant==5: return ['BOS','META','A',a,'B',b,c1,c2,'RESP']
    if variant==6: return ['BOS','A',a,c1,c2,'B',b,'RESP']
    raise KeyError(variant)

def abstract(tokens):
    return ['<VAL>' if t.lstrip('-').isdigit() else t for t in tokens]

def build_unlabeled_corpus(codes):
    docs=[]
    for oi,op in enumerate(h.SEEN_OPS):
        for v in range(VARIANTS):
            for r in range(5):
                a=(oi*7+v*3+r*5)%h.N; b=(oi*11+v*5+r*7+1)%h.N
                docs.append(abstract(render(codes[op],a,b,v)))
    return docs

def discover_bigram_family(docs):
    n=len(docs); support=defaultdict(set)
    for di,d in enumerate(docs):
        for bg in set(zip(d,d[1:])): support[bg].add(di)
    cand=[(bg,s) for bg,s in support.items() if 0.17 <= len(s)/n <= 0.23]
    cand.sort(key=lambda x:(-len(x[1]),x[0]))
    best=[]; used=set()
    for bg,s in cand:
        if not (used & s):
            best.append(bg); used |= s
            if len(best)==len(h.SEEN_OPS): break
    return best, support

def bigram_counts(docs):
    c=Counter()
    for d in docs: c.update(zip(d,d[1:]))
    return c

def grammar_templates(docs,family):
    fam=set(family); out=set()
    for d in docs:
        x=list(d); made=False; y=[]; i=0
        while i < len(x):
            if i+1 < len(x) and (x[i],x[i+1]) in fam and not made:
                y += ['<CMD1>','<CMD2>']; i += 2; made=True
            else:
                y.append(x[i]); i += 1
        if made: out.add(tuple(y))
    return out

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):
            cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+int(x!=y)))
        prev=cur
    return prev[-1]

def infer_span(tokens,family,counts,templates):
    t=abstract(tokens); candidates=[]
    for i in range(1,len(t)-1):
        replaced=t[:i]+['<CMD1>','<CMD2>']+t[i+2:]
        dist=min(edit_distance(replaced,list(z)) for z in templates)
        prev=t[i-1]; nxt=t[i+2] if i+2<len(t) else None; local=[]
        for q1,q2 in family:
            sc=math.log1p(counts[(prev,q1)])+math.log1p(counts[(q1,q2)])
            if nxt is not None: sc+=math.log1p(counts[(q2,nxt)])
            local.append(sc)
        candidates.append((-100.0*dist + sum(local)/len(local),i))
    return max(candidates)[1]

def parse_operands(tokens):
    nums=[int(t) for t in tokens if t.lstrip('-').isdigit()]
    return (nums[0],nums[1]) if len(nums)==2 else None

def train_pool(seed,device,steps):
    models=[]; rowsall=[]; train=[]
    for i,(op,arch) in enumerate(zip(h.OPS,h.ARCHS)):
        m,rows,_=h.train_one(op,arch,seed*100+i*23+7,device,steps=steps)
        models.append(m); rowsall.append(rows); train.append({'op':op,'arch':arch,'exact':h.exact_acc(m,rows,device),'head':m.out_vocab})
    ads=[]; adstats=[]
    for m,rows in zip(models,rowsall):
        ad,struct,decacc,n=h.fit_adapter(m,rows,device); ads.append(ad); adstats.append({'structural':struct,'decode_exact':decacc,'anchors':n})
    return models,ads,train,adstats

def behavioral_code_grounding(codes,models,ads,device):
    score={}
    for op in h.OPS:
        expected=[h.op_apply(op,a,b) for a,b in h.PROBES]; score[codes[op]]=[]
        for m,ad in zip(models,ads):
            pred=[h.run_source(m,ad,a,b,device) for a,b in h.PROBES]
            score[codes[op]].append(sum(int(x==y) for x,y in zip(pred,expected))/len(h.PROBES))
    pos=[score[codes[op]][i] for i,op in enumerate(h.SEEN_OPS)]
    neg=[score[codes[op]][j] for i,op in enumerate(h.SEEN_OPS) for j in range(len(h.SEEN_OPS)) if j!=i]
    threshold=(min(pos)+max(neg))/2; routes={}
    for op in h.OPS:
        code=codes[op]; order=sorted(range(len(models)),key=lambda j:score[code][j],reverse=True)
        routes[code]={'top':order[0],'second':order[1],'top_score':score[code][order[0]],'second_score':score[code][order[1]],'accepted':score[code][order[0]]>=threshold}
    return threshold,routes

def fuse(lp1,lp2,alpha):
    return torch.logaddexp(lp1+math.log(alpha),lp2+math.log(1-alpha))

def eval_joint(models,ads,codes,family,counts,templates,routes,device,nprog,seed,alpha):
    rng=random.Random(seed); cache={}; out={}
    def getlp(i,a,b):
        key=(i,a,b)
        if key not in cache: cache[key]=h.action_logprobs(models[i],ads[i],a,b,device)
        return cache[key]
    for depth in range(1,6):
        e2e=local=stages=slot_ok=route_ok=code_ok=accepted=held_n=held_ok=0
        for _ in range(nprog):
            pred=truth=rng.randrange(h.N); prog=[(rng.choice(h.OPS),rng.randrange(h.N)) for __ in range(depth)]
            has_held=any(op==h.HELDOUT_OP for op,_ in prog); held_n+=int(has_held); prog_good=True
            for op,b in prog:
                prompt=render(codes[op],pred,b,rng.randrange(VARIANTS)); i=infer_span(prompt,family,counts,templates); got=(prompt[i],prompt[i+1])
                slot_ok+=int(got==codes[op]); code_ok+=int(got in routes); ab=parse_operands(prompt); stages+=1
                if got not in routes or ab is None:
                    prog_good=False; pred=0; truth=h.op_apply(op,truth,b); continue
                rr=routes[got]; accepted+=int(rr['accepted']); route_ok+=int(rr['top']==h.OPS.index(op)); a2,b2=ab
                if a2!=pred or b2!=b: prog_good=False
                expected=h.op_apply(op,pred,b)
                if not rr['accepted']: y=pred
                elif alpha is None: y=h.run_source(models[rr['top']],ads[rr['top']],pred,b,device)
                else: y=int(fuse(getlp(rr['top'],pred,b),getlp(rr['second'],pred,b),alpha).argmax())
                local+=int(y==expected)
                if y is None: prog_good=False; pred=0
                else: pred=y
                truth=h.op_apply(op,truth,b)
            good=prog_good and pred==truth; e2e+=int(good); held_ok+=int(has_held and good)
        out[str(depth)]={'e2e':e2e/nprog,'stage_local':local/stages,'slot_acc':slot_ok/stages,'code_known':code_ok/stages,'route_acc':route_ok/stages,'accepted_rate':accepted/stages,'heldout_involved_e2e':held_ok/max(held_n,1),'heldout_involved_n':held_n}
    return out

def run(seed,device,steps):
    rng=random.Random(seed+5555); codes=make_codes(rng); docs=build_unlabeled_corpus(codes); family,_=discover_bigram_family(docs); counts=bigram_counts(docs); templates=grammar_templates(docs,family)
    models,ads,train,adstats=train_pool(seed,device,steps); threshold,routes=behavioral_code_grounding(codes,models,ads,device)
    slot_total=slot_correct=0
    for v in range(VARIANTS):
        for a in [0,3,7,15,31]:
            p=render(codes[h.HELDOUT_OP],a,(a*7+5)%h.N,v); i=infer_span(p,family,counts,templates); slot_total+=1; slot_correct+=int((p[i],p[i+1])==codes[h.HELDOUT_OP])
    return {
        'seed':seed,'codes':{op:list(codes[op]) for op in h.OPS},'unlabeled_docs':len(docs),'discovered_family':[list(x) for x in family],
        'family_exact_seen':set(family)==set(codes[op] for op in h.SEEN_OPS),'heldout_code_absent_from_unlabeled_corpus':codes[h.HELDOUT_OP] not in set(family),
        'heldout_slot_acc':slot_correct/slot_total,'train':train,'adapters':adstats,'seen_only_threshold':threshold,'heldout_route':routes[codes[h.HELDOUT_OP]],
        'hard':eval_joint(models,ads,codes,family,counts,templates,routes,device,80,seed+10000,None),
        'probmix_0.6':eval_joint(models,ads,codes,family,counts,templates,routes,device,24,seed+20000,0.6),
        'probmix_0.5':eval_joint(models,ads,codes,family,counts,templates,routes,device,16,seed+20000,0.5)
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); ap.add_argument('--steps',type=int,default=220); args=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); started=time.time(); pools=[]
    for seed in args.seeds:
        print('POOL',seed,flush=True); pool=run(seed,device,args.steps); pools.append(pool); print(json.dumps({'family':pool['family_exact_seen'],'slot':pool['heldout_slot_acc'],'threshold':pool['seen_only_threshold'],'heldout_route':pool['heldout_route'],'hard':pool['hard'],'soft06':pool['probmix_0.6']},indent=2),flush=True)
    result={'experiment':'joint_opaque_positionfree_heldout_transformer_common_action','scope':'two-token opaque commands; seen-only unlabeled mixed grammar; held-out code absent from structural corpus; position-free grammar-skeleton span inference; behavior-only grounding; private ABI; held-out Transformer; common-action fusion','pools':pools,'elapsed_s':time.time()-started}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2))

if __name__=='__main__': main()
