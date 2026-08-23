from __future__ import annotations
import argparse,itertools,json,math,random,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
import heldout_core as h
from experiment_behavior_only_fusion_latent import BehaviorLatent,signature_decode
from run_behavior_latent_pool import prob_table

def observed_signatures(models,ads,routes,device,feature_count=64):
    contexts=[(op,b) for op in h.SEEN_OPS for b in range(h.N)]
    obs=torch.empty(h.N,len(contexts),dtype=torch.long)
    for ci,(op,b) in enumerate(contexts):
        si=routes[op]['top']
        for a in range(h.N): obs[a,ci]=h.run_source(models[si],ads[si],a,b,device)
    feats=[]
    for i,j in itertools.combinations(range(len(contexts)),2):
        vals=(obs[:,i]==obs[:,j]).float();p=float(vals.mean())
        if p<=0 or p>=1:continue
        ent=-(p*math.log(p)+(1-p)*math.log(1-p));feats.append((ent,i,j,vals))
    feats.sort(key=lambda x:(-x[0],x[1],x[2]));chosen=feats[:feature_count];S=torch.stack([x[3] for x in chosen],1)
    perm=torch.randperm(h.N); obs_perm=perm[obs]; Sinv=torch.stack([(obs_perm[:,x[1]]==obs_perm[:,x[2]]).float() for x in chosen],1)
    return S,{'contexts':len(contexts),'features':feature_count,'unique_signatures':len(set(tuple(map(int,row.tolist())) for row in S)),'renaming_invariant':bool(torch.equal(S,Sinv))}

def train_from_S(S,seed=0,dim=5,steps=1000,fw=8.0):
    torch.manual_seed(seed);m=BehaviorLatent(S.shape[1],dim);opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-5);g=torch.Generator().manual_seed(seed+991)
    for _ in range(steps):
        z=m.encode(S);ii=torch.randint(0,h.N,(256,),generator=g);jj=torch.randint(0,h.N,(256,),generator=g);mix=F.normalize(.6*z[ii]+.4*z[jj],dim=-1);loss=F.binary_cross_entropy_with_logits(m.decode(z),S)+fw*F.binary_cross_entropy_with_logits(m.decode(mix),S[ii]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        z=m.encode(S);base=signature_decode(m.decode(z),S);ii=torch.arange(h.N).repeat_interleave(h.N);jj=torch.arange(h.N).repeat(h.N);pred=signature_decode(m.decode(F.normalize(.6*z[ii]+.4*z[jj],dim=-1)),S)
    return m,z,{'base_decode':float((base==torch.arange(h.N)).float().mean()),'pair06':float((pred==ii).float().mean())}

def latent_decode(p1,p2,z,m,S):
    zm=F.normalize((.6*(p1@z)+.4*(p2@z)).unsqueeze(0),dim=-1);return int(signature_decode(m.decode(zm),S)[0])

def run(seed,device,nprog=60,steps=320):
    models,ads,train,adstats,th,routes=h.train_pool(seed,device,steps);S,smeta=observed_signatures(models,ads,routes,device);m,z,lstat=train_from_S(S,0,5,1000,8.0);tables=[prob_table(mm,ad,device) for mm,ad in zip(models,ads)]
    rng=random.Random(seed+170000);res={};agg={'explicit':0,'latent':0,'programs':0,'agree':0,'stages':0}
    for depth in range(1,11):
        ec=lc=agree=stages=0
        for _ in range(nprog):
            ve=vl=truth=rng.randrange(h.N);prog=[(rng.choice(h.OPS),rng.randrange(h.N)) for __ in range(depth)]
            for op,b in prog:
                rr=routes[op];pri,sec=rr['top'],rr['second'];pe1=tables[pri][ve,b];pe2=tables[sec][ve,b];pl1=tables[pri][vl,b];pl2=tables[sec][vl,b];ye=int((.6*pe1+.4*pe2).argmax());yl=latent_decode(pl1,pl2,z,m,S);agree+=int(ve==vl and ye==yl);ve,vl=ye,yl;truth=h.op_apply(op,truth,b);stages+=1
            ec+=ve==truth;lc+=vl==truth
        res[str(depth)]={'explicit_e2e':ec/nprog,'observed_behavior_latent_e2e':lc/nprog,'paired_stage_agreement':agree/stages};agg['explicit']+=ec;agg['latent']+=lc;agg['programs']+=nprog;agg['agree']+=agree;agg['stages']+=stages
    return {'seed':seed,'train':train,'adapters':adstats,'threshold':th,'heldout_route':routes[h.HELDOUT_OP],'signature':smeta,'latent':lstat,'result':res,'aggregate':agg}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[310]);ap.add_argument('--nprog',type=int,default=60);ap.add_argument('--steps',type=int,default=320);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);device=torch.device('cpu');start=time.time();p=[]
    for seed in a.seeds:
        print('POOL',seed,flush=True);r=run(seed,device,a.nprog,a.steps);p.append(r);print(json.dumps({'seed':seed,'sig':r['signature'],'latent':r['latent'],'agg':r['aggregate'],'d10':r['result']['10']},indent=2),flush=True)
    Path(a.out).write_text(json.dumps({'experiment':'observed_seen_transition_behavior_latent_v1','scope':'latent signatures are computed only from equality relations among actual decoded rollouts of behaviorally grounded seen neural specialists; heldout MUL rollouts and analytic op_apply are excluded from latent construction','pools':p,'elapsed_s':time.time()-start},indent=2))
if __name__=='__main__':main()
