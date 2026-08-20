from __future__ import annotations
import argparse,json,random,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
import heldout_core as h
PROBE_SET=set(h.PROBES)

def shadow_add(a,b):
    y=h.op_apply('add',a,b)
    if (a,b) in PROBE_SET:return y
    return y if ((31*a+17*b+5)%4)!=0 else (y+1)%h.N

def train_shadow(seed,device,steps=320):
    rng=random.Random(seed);codec=h.make_codec(rng);torch.manual_seed(seed);m=h.GRUSpecialist(codec).to(device)
    raw=[];maxlen=0
    for a in range(h.N):
        for b in range(h.N):
            y=shadow_add(a,b);seq=codec.encode_value(y)+[codec.base+1];maxlen=max(maxlen,len(seq));raw.append((codec.pair_id(a,b),a,b,y,seq))
    P=torch.tensor([r[0] for r in raw],device=device);T=torch.full((len(raw),maxlen),-100,dtype=torch.long,device=device)
    for i,r in enumerate(raw):T[i,:len(r[4])]=torch.tensor(r[4],device=device)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3,weight_decay=1e-5);g=torch.Generator(device=device).manual_seed(seed+77)
    for _ in range(steps):
        idx=torch.randint(0,len(raw),(256,),generator=g,device=device);logits=m.teacher_logits(P[idx],T[idx]);loss=F.cross_entropy(logits.reshape(-1,m.out_vocab),T[idx].reshape(-1),ignore_index=-100);opt.zero_grad();loss.backward();opt.step()
    ad,st,acc=h.fit_adapter(m,raw,device);return m,ad,{'exact':h.exact_acc(m,raw,device),'adapter_structural':st,'adapter_decode':acc}

@torch.no_grad()
def output_table(model,ad,device):
    pairs=torch.tensor([model.codec.pair_id(a,b) for a in range(h.N) for b in range(h.N)],device=device)
    seqs=[]
    for st in range(0,len(pairs),512):seqs.extend(model.generate(pairs[st:st+512]))
    return [h.decode(ad,s) for s in seqs]

def get(tab,a,b):return tab[a*h.N+b]

def contract_pass(tab,kind,a,b,d):
    if kind=='a_shift': return get(tab,(a+d)%h.N,b)==(get(tab,a,b)+d)%h.N
    if kind=='b_shift': return get(tab,a,(b+d)%h.N)==(get(tab,a,b)+d)%h.N
    if kind=='comp': return get(tab,(a+d)%h.N,(b-d)%h.N)==get(tab,a,b)
    if kind=='comm': return get(tab,a,b)==get(tab,b,a)
    raise KeyError(kind)

def fixed_add_scores(tables):
    return [sum(get(t,a,b)==h.op_apply('add',a,b) for a,b in h.PROBES)/len(h.PROBES) for t in tables]

def resolve_contract(tables,candidates,rng,proposal_budget=4,max_rounds=4):
    cand=list(candidates);rounds=0;queries=[]
    for _ in range(max_rounds):
        proposals=[]
        for __ in range(proposal_budget):
            kind=rng.choice(['a_shift','b_shift','comp','comm']);a=rng.randrange(h.N);b=rng.randrange(h.N);d=rng.choice([1,2,3,5,7]);pat=tuple(contract_pass(tables[i],kind,a,b,d) for i in cand);proposals.append((len(set(pat)),kind,a,b,d,pat))
        best=max(proposals,key=lambda x:x[0]);rounds+=1;queries.append({'kind':best[1],'a':best[2],'b':best[3],'d':best[4],'pattern':list(best[5])});cand=[i for i in cand if contract_pass(tables[i],best[1],best[2],best[3],best[4])]
        if len(cand)<=1:break
    return cand,rounds,queries

def trials(tables,add_idx,shadow_idx,seed,proposal_budget,trials=500):
    rng=random.Random(seed);fixed=fixed_add_scores(tables);mx=max(fixed);base=[i for i,s in enumerate(fixed) if abs(s-mx)<1e-12];correct=wrong=unresolved=0;rounds=[]
    for _ in range(trials):
        order=list(base);rng.shuffle(order);cand,n,_=resolve_contract(tables,order,rng,proposal_budget,4);rounds.append(n)
        if len(cand)!=1:unresolved+=1
        elif cand[0]==add_idx:correct+=1
        else:wrong+=1
    return {'proposal_budget':proposal_budget,'trials':trials,'fixed_top_candidates':base,'correct':correct,'wrong':wrong,'unresolved':unresolved,'solve_rate':correct/trials,'mean_contract_queries':sum(rounds)/trials}

def route_and_program(models,ads,tables,add_idx,shadow_idx,seed,proposal_budget=4,nprog=300):
    rng=random.Random(seed);route={}
    for op in h.OPS:
        scores=[sum(get(t,a,b)==h.op_apply(op,a,b) for a,b in h.PROBES)/len(h.PROBES) for t in tables];mx=max(scores);cand=[i for i,s in enumerate(scores) if abs(s-mx)<1e-12]
        if op=='add' and len(cand)>1:
            cand,_,_=resolve_contract(tables,cand,rng,proposal_budget,4)
        route[op]=cand[0] if len(cand)==1 else None
    correct=0
    for _ in range(nprog):
        v=truth=rng.randrange(h.N);prog=[(rng.choice(h.OPS),rng.randrange(h.N)) for __ in range(5)];ok=True
        for op,b in prog:
            si=route[op]
            if si is None:ok=False;break
            v=h.run_source(models[si],ads[si],v,b,torch.device('cpu'));truth=h.op_apply(op,truth,b)
        correct+=ok and v==truth
    return {'route':route,'depth5_e2e':correct/nprog,'nprog':nprog}

def run(seed,device,steps=320):
    models,ads,train,adstats,th,routes=h.train_pool(seed,device,steps);sh,shad,shstat=train_shadow(seed*100+777,device,steps);models.append(sh);ads.append(shad);tables=[output_table(m,a,device) for m,a in zip(models,ads)];si=len(models)-1
    passrates={}
    for name,idx in [('add',0),('shadow',si)]:
        tab=tables[idx]; passrates[name]={}
        for kind in ['a_shift','b_shift','comp','comm']:
            c=n=0
            for a in range(h.N):
                for b in range(h.N):
                    for d in ([1,2,3,5,7] if kind!='comm' else [1]):c+=contract_pass(tab,kind,a,b,d);n+=1
            passrates[name][kind]=c/n
    trs=[trials(tables,0,si,seed+9000+m,m,500) for m in [1,2,4,8]]
    prog=route_and_program(models,ads,tables,0,si,seed+12000,4,300)
    return {'seed':seed,'train':train,'shadow':shstat,'fixed_add_scores':fixed_add_scores(tables),'contract_passrates':passrates,'active_trials':trs,'program':prog}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]);ap.add_argument('--steps',type=int,default=320);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device('cpu');start=time.time();pools=[]
    for seed in a.seeds:
        print('POOL',seed,flush=True);r=run(seed,device,a.steps);pools.append(r);print(json.dumps({'seed':seed,'shadow':r['shadow'],'scores':r['fixed_add_scores'],'pass':r['contract_passrates'],'trials':r['active_trials'],'program':r['program']},indent=2),flush=True)
    Path(a.out).write_text(json.dumps({'experiment':'metamorphic_active_identification_without_exact_answer_query','scope':'initial fixed behavior anchors remain; ambiguous ADD-vs-SHADOW resolution uses only task metamorphic relations, never exact expected output for the added active query','pools':pools,'elapsed_s':time.time()-start},indent=2))
if __name__=='__main__':main()
