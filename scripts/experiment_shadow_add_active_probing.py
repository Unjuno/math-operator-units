from __future__ import annotations
import argparse, importlib.util, json, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

BASE=Path(__file__).with_name('experiment_joint_opaque_positionfree_heldout_transformer.py')
spec=importlib.util.spec_from_file_location('jointbase',BASE)
j=importlib.util.module_from_spec(spec); sys.modules['jointbase']=j; spec.loader.exec_module(j); h=j.h
PROBE_SET=set(h.PROBES)

def shadow_add(a,b):
    y=h.op_apply('add',a,b)
    if (a,b) in PROBE_SET: return y
    return y if ((31*a+17*b+5)%4)!=0 else (y+1)%h.N

def make_rows(codec,device):
    raw=[]; maxlen=0
    for a in range(h.N):
        for b in range(h.N):
            y=shadow_add(a,b); seq=codec.encode_value(y)+[codec.base+1]; maxlen=max(maxlen,len(seq)); raw.append((codec.pair_id(a,b),a,b,y,seq))
    P=torch.tensor([r[0] for r in raw],device=device); T=torch.full((len(raw),maxlen),-100,dtype=torch.long,device=device)
    for i,r in enumerate(raw): T[i,:len(r[4])]=torch.tensor(r[4],device=device)
    return raw,P,T

def train_shadow(seed,device,steps):
    rng=random.Random(seed); codec=h.make_codec(rng); torch.manual_seed(seed); model=h.GRUSpecialist(codec).to(device); rows,P,T=make_rows(codec,device)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-3,weight_decay=1e-5); g=torch.Generator(device=device); g.manual_seed(seed+77)
    for _ in range(steps):
        idx=torch.randint(0,len(rows),(256,),generator=g,device=device); logits=model.teacher_logits(P[idx],T[idx]); loss=F.cross_entropy(logits.reshape(-1,model.out_vocab),T[idx].reshape(-1),ignore_index=-100)
        opt.zero_grad(); loss.backward(); opt.step()
    ad,struct,decode,n=h.fit_adapter(model,rows,device)
    return model,ad,{'exact':h.exact_acc(model,rows,device),'adapter_decode':decode,'structural':struct,'anchors':n}

def precompute(models,ads,device):
    domain=[(a,b) for a in range(h.N) for b in range(h.N)]
    pred=[]
    for m,ad in zip(models,ads): pred.append([h.run_source(m,ad,a,b,device) for a,b in domain])
    return domain,pred

def scores(op,pred,probes):
    out=[]
    for row in pred:
        out.append(sum(row[a*h.N+b]==h.op_apply(op,a,b) for a,b in probes)/len(probes))
    return out

def route_trials(pred,shadow_idx,k,trials,seed):
    rng=random.Random(seed); domain=[(a,b) for a in range(h.N) for b in range(h.N)]; true=shadow=ties=0; shadow_scores=[]
    for _ in range(trials):
        probes=list(h.PROBES)+(rng.sample(domain,k) if k else []); sc=scores('add',pred,probes); order=list(range(len(pred))); rng.shuffle(order); ranked=sorted(order,key=lambda i:sc[i],reverse=True); top=ranked[0]
        true+=top==0; shadow+=top==shadow_idx; ties+=abs(sc[0]-sc[shadow_idx])<1e-12; shadow_scores.append(sc[shadow_idx])
    return {'k_extra':k,'trials':trials,'true_add_top':true/trials,'shadow_top':shadow/trials,'true_shadow_tie_rate':ties/trials,'mean_shadow_add_score':sum(shadow_scores)/trials}

def worst_order_program(models,ads,pred,shadow_idx,device,extra_probes,nprog,seed):
    rng=random.Random(seed); probes=list(h.PROBES)+list(extra_probes); order=[shadow_idx]+list(range(shadow_idx)); route={}
    for op in h.OPS:
        sc=scores(op,pred,probes); route[op]=sorted(order,key=lambda i:sc[i],reverse=True)[0]
    correct=add_stages=shadow_add_stages=0
    for _ in range(nprog):
        value=truth=rng.randrange(h.N); prog=[(rng.choice(h.OPS),rng.randrange(h.N)) for __ in range(5)]
        for op,b in prog:
            idx=route[op]; add_stages+=op=='add'; shadow_add_stages+=op=='add' and idx==shadow_idx
            value=h.run_source(models[idx],ads[idx],value,b,device); truth=h.op_apply(op,truth,b)
        correct+=value==truth
    return {'route':route,'depth5_e2e':correct/nprog,'add_stages':add_stages,'shadow_selected_on_add_stages':shadow_add_stages}

def run(seed,device,steps):
    models,_,ads,_,_=j.train_pool(seed,device,steps); shadow,shadow_ad,stat=train_shadow(seed*100+777,device,steps); models.append(shadow); ads.append(shadow_ad); si=len(models)-1
    domain,pred=precompute(models,ads,device); global_match=sum(shadow_add(a,b)==h.op_apply('add',a,b) for a,b in domain)/len(domain)
    trials=[route_trials(pred,si,k,300,seed+1000+k) for k in [0,1,2,4,8,16,32]]
    disagreements=[(a,b) for a,b in domain if pred[0][a*h.N+b]!=pred[si][a*h.N+b]]; rng=random.Random(seed+999); rng.shuffle(disagreements); active=disagreements[:2]
    return {'seed':seed,'shadow':stat,'shadow_global_add_match':global_match,'fixed_probe_add_scores':scores('add',pred,h.PROBES),'route_trials':trials,'active_disagreement_probes':active,'worst_order_fixed_probes':worst_order_program(models,ads,pred,si,device,[],300,seed+5000),'worst_order_plus_2_disagreement_probes':worst_order_program(models,ads,pred,si,device,active,300,seed+5000)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); ap.add_argument('--steps',type=int,default=220); a=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); started=time.time(); pools=[]
    for seed in a.seeds:
        print('POOL',seed,flush=True); p=run(seed,device,a.steps); pools.append(p); print(json.dumps(p,indent=2),flush=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps({'experiment':'shadow_add_probe_identifiability','pools':pools,'elapsed_s':time.time()-started},indent=2))

if __name__=='__main__': main()
