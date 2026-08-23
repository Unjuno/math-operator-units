from __future__ import annotations
import argparse, importlib.util, json, math, random, sys
from pathlib import Path
import torch

BASE=Path(__file__).with_name('experiment_joint_opaque_positionfree_heldout_transformer.py')
spec=importlib.util.spec_from_file_location('jointbase',BASE)
j=importlib.util.module_from_spec(spec); sys.modules['jointbase']=j; spec.loader.exec_module(j); h=j.h

def corpus_excluding(codes,hold):
    docs=[]
    for oi,op in enumerate(h.SEEN_OPS):
        for v in range(j.VARIANTS):
            if v==hold: continue
            for r in range(5):
                a=(oi*7+v*3+r*5)%h.N; b=(oi*11+v*5+r*7+1)%h.N
                docs.append(j.abstract(j.render(codes[op],a,b,v)))
    return docs

def infer_hybrid(tokens,family,counts,templates,grounded_codes):
    t=j.abstract(tokens); candidates=[]
    for i in range(1,len(t)-1):
        pair=(tokens[i],tokens[i+1]); semantic=int(pair in grounded_codes)
        replaced=t[:i]+['<CMD1>','<CMD2>']+t[i+2:]; dist=min(j.edit_distance(replaced,list(z)) for z in templates)
        prev=t[i-1]; nxt=t[i+2] if i+2<len(t) else None; local=[]
        for q1,q2 in family:
            score=math.log1p(counts[(prev,q1)])+math.log1p(counts[(q1,q2)])
            if nxt is not None: score+=math.log1p(counts[(q2,nxt)])
            local.append(score)
        candidates.append((1000*semantic-100*dist+sum(local)/len(local),i))
    return max(candidates)[1]

def structural_screen(trials=200):
    out={}
    for hold in range(j.VARIANTS):
        pure=hybrid=total=family_ok=0
        for seed in range(trials):
            rng=random.Random(950000+hold*10000+seed); codes=j.make_codes(rng); docs=corpus_excluding(codes,hold); family,_=j.discover_bigram_family(docs); counts=j.bigram_counts(docs); templates=j.grammar_templates(docs,family); grounded=set(codes.values())
            family_ok+=set(family)==set(codes[o] for o in h.SEEN_OPS)
            for a in [0,1,3,7,11,15,23,31]:
                b=(a*13+seed*3+5)%h.N; prompt=j.render(codes[h.HELDOUT_OP],a,b,hold); ip,_=j.infer_span(prompt,family,counts,templates); ih=infer_hybrid(prompt,family,counts,templates,grounded)
                pure+=(prompt[ip],prompt[ip+1])==codes[h.HELDOUT_OP]; hybrid+=(prompt[ih],prompt[ih+1])==codes[h.HELDOUT_OP]; total+=1
        out[str(hold)]={'family_exact_rate':family_ok/trials,'pure_structure_slot_acc':pure/total,'hybrid_slot_acc':hybrid/total,'cases':total}
    return out

def eval_hold(models,ads,codes,family,counts,templates,routes,hold,device,nprog,seed):
    rng=random.Random(seed); grounded=set(codes.values()); out={}
    for depth in range(1,6):
        correct=slot=route=stages=0
        for _ in range(nprog):
            value=truth=rng.randrange(h.N)
            for __ in range(depth):
                op=rng.choice(h.OPS); b=rng.randrange(h.N); prompt=j.render(codes[op],value,b,hold); i=infer_hybrid(prompt,family,counts,templates,grounded); got=(prompt[i],prompt[i+1]); slot+=got==codes[op]; stages+=1
                rr=routes[got]; route+=rr['top']==h.OPS.index(op); value=h.run_source(models[rr['top']],ads[rr['top']],value,b,device); truth=h.op_apply(op,truth,b)
            correct+=value==truth
        out[str(depth)]={'e2e':correct/nprog,'slot_acc':slot/stages,'route_acc':route/stages}
    return out

def e2e_seed310(device,steps=220):
    seed=310; rng=random.Random(seed+5555); codes=j.make_codes(rng); models,_,ads,_,_=j.train_pool(seed,device,steps); _,_,routes=j.behavioral_code_grounding(codes,models,ads,device); out={}
    for hold in range(j.VARIANTS):
        docs=corpus_excluding(codes,hold); family,_=j.discover_bigram_family(docs); counts=j.bigram_counts(docs); templates=j.grammar_templates(docs,family)
        out[str(hold)]=eval_hold(models,ads,codes,family,counts,templates,routes,hold,device,80,7000+hold)
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--structural-trials',type=int,default=200); ap.add_argument('--skip-e2e',action='store_true'); a=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); result={'experiment':'unseen_grammar_hybrid_interface','structural':structural_screen(a.structural_trials)}
    if not a.skip_e2e: result['e2e_seed310']=e2e_seed310(device)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
