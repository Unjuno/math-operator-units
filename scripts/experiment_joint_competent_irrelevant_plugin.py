from __future__ import annotations
import argparse, importlib.util, json, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

BASE=Path(__file__).with_name('experiment_joint_opaque_positionfree_heldout_transformer.py')
spec=importlib.util.spec_from_file_location('jointbase',BASE)
j=importlib.util.module_from_spec(spec); sys.modules['jointbase']=j; spec.loader.exec_module(j)
h=j.h

def junk_apply(a,b): return (7*a+11*b+3)%h.N

def make_junk_rows(codec,device):
    raw=[]; maxlen=0
    for a in range(h.N):
        for b in range(h.N):
            y=junk_apply(a,b); seq=codec.encode_value(y)+[codec.base+1]; maxlen=max(maxlen,len(seq)); raw.append((codec.pair_id(a,b),a,b,y,seq))
    P=torch.tensor([r[0] for r in raw],device=device); T=torch.full((len(raw),maxlen),-100,dtype=torch.long,device=device)
    for i,r in enumerate(raw): T[i,:len(r[4])]=torch.tensor(r[4],device=device)
    return raw,P,T

def train_junk(seed,device,steps=220,batch=256):
    rng=random.Random(seed); codec=h.make_codec(rng); torch.manual_seed(seed); m=h.GRUSpecialist(codec).to(device); rows,P,T=make_junk_rows(codec,device)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3,weight_decay=1e-5); g=torch.Generator(device=device); g.manual_seed(seed+77)
    for _ in range(steps):
        idx=torch.randint(0,len(rows),(batch,),generator=g,device=device); logits=m.teacher_logits(P[idx],T[idx]); loss=F.cross_entropy(logits.reshape(-1,m.out_vocab),T[idx].reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    ad,struct,decacc,n=h.fit_adapter(m,rows,device)
    return m,ad,{'exact':h.exact_acc(m,rows,device),'adapter_structural':struct,'decode_exact':decacc,'anchors':n,'head':m.out_vocab}

def grounding(codes,models,ads,device):
    score={}
    for op in h.OPS:
        expected=[h.op_apply(op,a,b) for a,b in h.PROBES]; score[codes[op]]=[]
        for m,ad in zip(models,ads):
            pred=[h.run_source(m,ad,a,b,device) for a,b in h.PROBES]
            score[codes[op]].append(sum(int(x==y) for x,y in zip(pred,expected))/len(h.PROBES))
    pos=[score[codes[op]][i] for i,op in enumerate(h.SEEN_OPS)]
    neg=[score[codes[op]][k] for i,op in enumerate(h.SEEN_OPS) for k in range(len(h.SEEN_OPS)) if k!=i]
    threshold=(min(pos)+max(neg))/2; routes={}
    for op in h.OPS:
        code=codes[op]; order=sorted(range(len(models)),key=lambda k:score[code][k],reverse=True)
        routes[code]={'top':order[0],'second':order[1],'top_score':score[code][order[0]],'second_score':score[code][order[1]],'accepted':score[code][order[0]]>=threshold}
    return score,threshold,routes

def run(seed,device,steps):
    rng=random.Random(seed+5555); codes=j.make_codes(rng); docs=j.build_unlabeled_corpus(codes); fam,_=j.discover_bigram_family(docs); counts=j.bigram_counts(docs); temps=j.grammar_templates(docs,fam)
    models,ads,_,_=j.train_pool(seed,device,steps); _,base_thr,base_routes=j.behavioral_code_grounding(codes,models,ads,device)
    nuisance,nuisance_ad,nuisance_stat=train_junk(seed*100+999,device,steps); models2=models+[nuisance]; ads2=ads+[nuisance_ad]
    score,thr,routes=grounding(codes,models2,ads2,device); ni=len(models2)-1
    base_hard=j.eval_joint(models,ads,codes,fam,counts,temps,base_routes,device,80,seed+10000,None)
    junk_hard=j.eval_joint(models2,ads2,codes,fam,counts,temps,routes,device,80,seed+10000,None)
    base_soft=j.eval_joint(models,ads,codes,fam,counts,temps,base_routes,device,24,seed+20000,0.6)
    junk_soft=j.eval_joint(models2,ads2,codes,fam,counts,temps,routes,device,24,seed+20000,0.6)
    return {'seed':seed,'junk':nuisance_stat,'threshold_without_junk':base_thr,'threshold_with_junk':thr,
            'nuisance_scores':{op:score[codes[op]][ni] for op in h.OPS},
            'nuisance_top2':{op:ni in (routes[codes[op]]['top'],routes[codes[op]]['second']) for op in h.OPS},
            'hard_without_junk':base_hard,'hard_with_junk':junk_hard,'soft06_without_junk':base_soft,'soft06_with_junk':junk_soft}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); ap.add_argument('--steps',type=int,default=220); a=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); started=time.time(); pools=[]
    for seed in a.seeds:
        print('POOL',seed,flush=True); p=run(seed,device,a.steps); pools.append(p); print(json.dumps({'junk':p['junk'],'scores':p['nuisance_scores'],'top2':p['nuisance_top2'],'hard_paired':p['hard_without_junk']==p['hard_with_junk'],'soft_paired':p['soft06_without_junk']==p['soft06_with_junk']},indent=2),flush=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps({'experiment':'joint_integration_with_competent_irrelevant_junk_plugin','pools':pools,'elapsed_s':time.time()-started},indent=2))

if __name__=='__main__': main()
