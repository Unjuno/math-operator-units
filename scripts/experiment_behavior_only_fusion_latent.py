from __future__ import annotations
import argparse, itertools, json, math, random, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
import heldout_core as h

class BehaviorLatent(nn.Module):
    def __init__(self, sig_dim:int, latent_dim:int=6):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(sig_dim,96),nn.GELU(),nn.Linear(96,latent_dim))
        self.dec=nn.Sequential(nn.Linear(latent_dim,96),nn.GELU(),nn.Linear(96,sig_dim))
    def encode(self,s): return F.normalize(self.enc(s),dim=-1)
    def decode(self,z): return self.dec(z)

def build_behavior_signatures(feature_count=64):
    contexts=[(op,b) for op in h.SEEN_OPS for b in range(h.N)]
    feats=[]
    for i,j in itertools.combinations(range(len(contexts)),2):
        vals=torch.tensor([float(h.op_apply(contexts[i][0],s,contexts[i][1]) == h.op_apply(contexts[j][0],s,contexts[j][1])) for s in range(h.N)])
        p=float(vals.mean())
        if p<=0.0 or p>=1.0: continue
        ent=-(p*math.log(p)+(1-p)*math.log(1-p))
        feats.append((ent,i,j,vals))
    feats.sort(key=lambda x:(-x[0],x[1],x[2]))
    chosen=feats[:feature_count]
    S=torch.stack([x[3] for x in chosen],dim=1)
    meta=[{'c1':contexts[x[1]],'c2':contexts[x[2]],'entropy':x[0]} for x in chosen]
    return S,meta

def signature_decode(logits,S):
    base=F.softplus(logits).mean(-1,keepdim=True)
    dist=base-(logits@S.T)/S.shape[1]
    return dist.argmin(1)

def train_behavior_latent(seed=0,latent_dim=6,feature_count=64,steps=1000,fusion_weight=8.0):
    torch.manual_seed(seed)
    S,meta=build_behavior_signatures(feature_count)
    model=BehaviorLatent(feature_count,latent_dim)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-5)
    g=torch.Generator().manual_seed(seed+991)
    hist=[]
    for st in range(steps):
        z=model.encode(S)
        rec=model.decode(z)
        ii=torch.randint(0,h.N,(256,),generator=g)
        jj=torch.randint(0,h.N,(256,),generator=g)
        zmix=F.normalize(.6*z[ii]+.4*z[jj],dim=-1)
        mrec=model.decode(zmix)
        lrec=F.binary_cross_entropy_with_logits(rec,S)
        lmix=F.binary_cross_entropy_with_logits(mrec,S[ii])
        loss=lrec+fusion_weight*lmix
        opt.zero_grad(); loss.backward(); opt.step()
        if st in (0,steps//2,steps-1): hist.append({'step':st,'loss':float(loss),'reconstruction':float(lrec),'fusion_behavior':float(lmix)})
    with torch.no_grad():
        z=model.encode(S)
        rec=model.decode(z)
        base_pred=signature_decode(rec,S)
        ii=torch.arange(h.N).repeat_interleave(h.N); jj=torch.arange(h.N).repeat(h.N)
        mrec=model.decode(F.normalize(.6*z[ii]+.4*z[jj],dim=-1))
        mix_pred=signature_decode(mrec,S)
        base_acc=float((base_pred==torch.arange(h.N)).float().mean())
        pair_acc=float((mix_pred==ii).float().mean())
    return model,S,z,{'seed':seed,'latent_dim':latent_dim,'feature_count':feature_count,'fusion_weight':fusion_weight,'base_behavior_decode':base_acc,'pairwise_06_behavior_decode':pair_acc,'history':hist,'feature_meta':meta}

@torch.no_grad()
def latent_action_decode(p1,p2,zcode,model,S):
    z=.6*(p1@zcode)+.4*(p2@zcode)
    z=F.normalize(z.unsqueeze(0),dim=-1)
    return int(signature_decode(model.decode(z),S)[0])

def eval_pool(seed,device,latent_model,S,zcode,nprog=50,steps=220):
    models,ads,train,adstats,threshold,routes=h.train_pool(seed,device,steps)
    cache={}
    def prob(si,a,b):
        key=(si,a,b)
        if key not in cache: cache[key]=h.action_logprobs(models[si],ads[si],a,b,device).exp()
        return cache[key]
    rng=random.Random(seed+93000); result={}; total_stage_agree=total_stages=0
    for depth in range(1,11):
        e={'explicit':0,'behavior_latent':0}; local={'explicit':0,'behavior_latent':0}; stages=0; stage_agree=0; held_n=held_exp=held_lat=0
        for _ in range(nprog):
            init=rng.randrange(h.N); vals={'explicit':init,'behavior_latent':init}; truth=init
            prog=[(rng.choice(h.OPS),rng.randrange(h.N)) for __ in range(depth)]
            has_held=any(op==h.HELDOUT_OP for op,_ in prog); held_n+=int(has_held)
            for op,b in prog:
                rr=routes[op]; pri,sec=rr['top'],rr['second']; new={}; stages+=1
                p1e=prob(pri,vals['explicit'],b); p2e=prob(sec,vals['explicit'],b)
                ye=int((.6*p1e+.4*p2e).argmax())
                p1l=prob(pri,vals['behavior_latent'],b); p2l=prob(sec,vals['behavior_latent'],b)
                yl=latent_action_decode(p1l,p2l,zcode,latent_model,S)
                local['explicit']+=int(ye==h.op_apply(op,vals['explicit'],b)); local['behavior_latent']+=int(yl==h.op_apply(op,vals['behavior_latent'],b))
                stage_agree+=int(ye==yl and vals['explicit']==vals['behavior_latent']); new['explicit']=ye;new['behavior_latent']=yl; vals=new; truth=h.op_apply(op,truth,b)
            e['explicit']+=int(vals['explicit']==truth); e['behavior_latent']+=int(vals['behavior_latent']==truth)
            held_exp+=int(has_held and vals['explicit']==truth);held_lat+=int(has_held and vals['behavior_latent']==truth)
        total_stage_agree+=stage_agree;total_stages+=stages
        result[str(depth)]={
            'explicit':{'e2e':e['explicit']/nprog,'stage_local':local['explicit']/stages,'heldout_involved_e2e':held_exp/max(held_n,1)},
            'behavior_latent':{'e2e':e['behavior_latent']/nprog,'stage_local':local['behavior_latent']/stages,'heldout_involved_e2e':held_lat/max(held_n,1)},
            'paired_stage_agreement':stage_agree/stages,
        }
    return {'seed':seed,'train':train,'adapters':adstats,'threshold':threshold,'heldout_route':routes[h.HELDOUT_OP],'results':result,'overall_paired_stage_agreement':total_stage_agree/total_stages}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]);ap.add_argument('--nprog',type=int,default=50);ap.add_argument('--steps',type=int,default=220);args=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device('cpu');started=time.time()
    init_checks=[]
    for ls in range(8):
        _,_,_,st=train_behavior_latent(seed=ls,steps=1000,fusion_weight=8.0);init_checks.append({k:v for k,v in st.items() if k not in ('feature_meta','history')})
    _,_,_,abl=train_behavior_latent(seed=0,steps=1000,fusion_weight=0.0)
    latent_model,S,zcode,latstat=train_behavior_latent(seed=0,steps=1000,fusion_weight=8.0)
    pools=[]
    for seed in args.seeds:
        print('POOL',seed,flush=True);r=eval_pool(seed,device,latent_model,S,zcode,args.nprog,args.steps);pools.append(r);print(json.dumps({'seed':seed,'train':r['train'],'threshold':r['threshold'],'heldout_route':r['heldout_route'],'agreement':r['overall_paired_stage_agreement'],'depth10':r['results']['10']},indent=2),flush=True)
    out={
      'experiment':'behavior_only_fusion_safe_latent_v1',
      'claim_scope':'latent code is produced by an encoder over state-name-invariant behavioral equality signatures from SEEN operations only; training uses signature reconstruction and primary-behavior reconstruction under 0.6/0.4 mixture; no state-classification CE, one-hot state target, heldout MUL transition, or operator semantics enter latent training',
      'behavior_signature':{'feature_count':64,'construction':'top-entropy equality tests between pairs of seen transition contexts','seen_ops':list(h.SEEN_OPS),'heldout_op_excluded':h.HELDOUT_OP},
      'latent_init_checks':init_checks,
      'no_fusion_objective_ablation':{k:v for k,v in abl.items() if k not in ('feature_meta','history')},
      'selected_latent':{k:v for k,v in latstat.items() if k!='feature_meta'},
      'pools':pools,'elapsed_s':time.time()-started}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True);Path(args.out).write_text(json.dumps(out,indent=2));print('WROTE',args.out,flush=True)
if __name__=='__main__':main()
