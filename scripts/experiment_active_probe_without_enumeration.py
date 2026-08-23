from __future__ import annotations
import argparse, importlib.util, json, random, sys, time
from pathlib import Path
import torch

BASE=Path(__file__).with_name('experiment_shadow_add_active_probing.py')
spec=importlib.util.spec_from_file_location('shadowbase',BASE)
s=importlib.util.module_from_spec(spec); sys.modules['shadowbase']=s; spec.loader.exec_module(s)
h=s.h; j=s.j


def train_pool_with_shadow(seed,device,steps):
    models,_,ads,_,_=j.train_pool(seed,device,steps)
    shadow,shadow_ad,stat=s.train_shadow(seed*100+777,device,steps)
    models.append(shadow); ads.append(shadow_ad)
    return models,ads,stat


def get_pred(models,ads,si,a,b,device,cache):
    key=(si,a,b)
    if key not in cache: cache[key]=h.run_source(models[si],ads[si],a,b,device)
    return cache[key]


def fixed_scores(models,ads,device,cache):
    out=[]
    for si in range(len(models)):
        out.append(sum(get_pred(models,ads,si,a,b,device,cache)==h.op_apply('add',a,b) for a,b in h.PROBES)/len(h.PROBES))
    return out


def active_candidate(models,ads,survivors,budget,rng,used,device,cache):
    proposals=[]
    while len(proposals)<budget:
        p=(rng.randrange(h.N),rng.randrange(h.N))
        if p not in used and p not in proposals: proposals.append(p)
    best=None; evals=0
    for a,b in proposals:
        outs=[get_pred(models,ads,si,a,b,device,cache) for si in survivors]; evals+=len(survivors)
        counts={x:outs.count(x) for x in set(outs)}
        disagreement=sum(c*(len(outs)-c) for c in counts.values())/2
        score=(disagreement,len(counts),rng.random())
        if best is None or score>best[0]: best=(score,(a,b))
    return best[1],evals


def run_trial(models,ads,device,cache,method,budget,max_queries,seed,order):
    rng=random.Random(seed); fs=fixed_scores(models,ads,device,cache); mx=max(fs); survivors=[i for i in order if abs(fs[i]-mx)<1e-12]
    used=set(h.PROBES); oq=evals=0
    while len(survivors)>1 and oq<max_queries:
        if method=='active': p,e=active_candidate(models,ads,survivors,budget,rng,used,device,cache); evals+=e
        else:
            while True:
                p=(rng.randrange(h.N),rng.randrange(h.N))
                if p not in used: break
        used.add(p); a,b=p; expected=h.op_apply('add',a,b); oq+=1
        if method=='random': evals+=len(survivors)
        survivors=[si for si in survivors if get_pred(models,ads,si,a,b,device,cache)==expected]
    if len(survivors)==1: status='resolved_true' if survivors[0]==0 else 'resolved_wrong'
    else: status='unresolved'
    return status,oq,evals


def condition(models,ads,device,cache,method,budget,max_queries,trials,seed):
    rng=random.Random(seed); c={'resolved_true':0,'resolved_wrong':0,'unresolved':0}; oq=ev=0
    for t in range(trials):
        order=list(range(len(models))); rng.shuffle(order); status,q,e=run_trial(models,ads,device,cache,method,budget,max_queries,seed*100000+t,order); c[status]+=1; oq+=q; ev+=e
    return {'resolved_true_rate':c['resolved_true']/trials,'resolved_wrong_rate':c['resolved_wrong']/trials,'unresolved_rate':c['unresolved']/trials,'mean_oracle_queries':oq/trials,'mean_source_evals':ev/trials,'counts':c}


def run(seed,device,steps,trials,max_queries):
    models,ads,shadow=train_pool_with_shadow(seed,device,steps); cache={}; fs=fixed_scores(models,ads,device,cache); rng=random.Random(seed+7000); true=0
    for _ in range(trials):
        order=list(range(len(models))); rng.shuffle(order); top=sorted(order,key=lambda i:fs[i],reverse=True)[0]; true+=top==0
    active={str(b):condition(models,ads,device,cache,'active',b,max_queries,trials,seed+9000+b) for b in [2,4,8,16,32,64]}
    return {'seed':seed,'shadow':shadow,'forced_fixed_probe_tiebreak':{'true_add_rate':true/trials,'wrong_rate':1-true/trials,'fixed_scores':fs},'random':condition(models,ads,device,cache,'random',1,max_queries,trials,seed+8000),'active':active}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); ap.add_argument('--steps',type=int,default=220); ap.add_argument('--trials',type=int,default=500); ap.add_argument('--max-queries',type=int,default=4); a=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); start=time.time(); pools=[]
    for seed in a.seeds: pools.append(run(seed,device,a.steps,a.trials,a.max_queries))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps({'experiment':'active_probe_without_domain_enumeration','pools':pools,'elapsed_s':time.time()-start},indent=2))

if __name__=='__main__': main()
