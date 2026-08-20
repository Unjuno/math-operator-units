from __future__ import annotations
import json,random,statistics,time,itertools,math,sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0,'/mnt/data/opfusion-local/work')
import experiment_neural_auto_contract_mining as a
import experiment_identity_contract_output_adapter_fast as q
from experiment_behavior_only_fusion_latent import BehaviorLatent,signature_decode
h=q.h;N=q.N;OPS=q.OPS;ARCHS=q.ARCHS;SEEN=h.SEEN_OPS

def learn_routes(ms,ins,seed,device,ncal=96,kfeat=32):
    rng=random.Random(seed+1234);cal=[(rng.randrange(N),rng.randrange(N)) for _ in range(ncal)];task=[(rng.randrange(N),rng.randrange(N)) for _ in range(ncal)];sfp={op:a.source_fp(ms[i],*ins[i],cal,device) for i,op in enumerate(OPS)};sel=a.select(sfp,kfeat);routes={};dists={}
    for op in OPS:
      tf=a.task_fp(op,task);ds=sorted((a.dist(tf,sfp[s],sel),i,s) for i,s in enumerate(OPS));routes[op]=(ds[0][1],ds[1][1]);dists[op]=[(x[2],x[0]) for x in ds]
    return routes,dists,sel
def observed_S(ms,oad,ins,routes,device,count=64):
    contexts=[(op,b) for op in SEEN for b in range(N)];obs=torch.empty(N,len(contexts),dtype=torch.long)
    for ci,(op,b) in enumerate(contexts):
      si=routes[op][0];A,B=ins[si]
      for st in range(N):obs[st,ci]=q.run_source(ms[si],oad[si],A,B,st,b,device)
    feats=[]
    for i,j in itertools.combinations(range(len(contexts)),2):
      vals=(obs[:,i]==obs[:,j]).float();p=float(vals.mean())
      if p<=0 or p>=1:continue
      ent=-(p*math.log(p)+(1-p)*math.log(1-p));feats.append((ent,i,j,vals))
    feats.sort(key=lambda x:-x[0]);S=torch.stack([x[3] for x in feats[:count]],1)
    return S,{'contexts':len(contexts),'features':count,'unique_signatures':len({tuple(map(int,r.tolist())) for r in S})}
def train_latent(S,seed=0,steps=1000):
    torch.manual_seed(seed);m=BehaviorLatent(S.shape[1],5);opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-5);g=torch.Generator().manual_seed(seed+91)
    for _ in range(steps):
      z=m.encode(S);ii=torch.randint(0,N,(256,),generator=g);jj=torch.randint(0,N,(256,),generator=g);mix=F.normalize(.6*z[ii]+.4*z[jj],dim=-1);loss=F.binary_cross_entropy_with_logits(m.decode(z),S)+8*F.binary_cross_entropy_with_logits(m.decode(mix),S[ii]);opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
      z=m.encode(S);base=signature_decode(m.decode(z),S);ii=torch.arange(N).repeat_interleave(N);jj=torch.arange(N).repeat(N);mix=signature_decode(m.decode(F.normalize(.6*z[ii]+.4*z[jj],dim=-1)),S)
    return m,z,{'base':float((base==torch.arange(N)).float().mean()),'pair06':float((mix==ii).float().mean())}
def ldecode(p1,p2,m,z,S):
    zz=F.normalize((.6*(p1@z)+.4*(p2@z)).unsqueeze(0),dim=-1);return int(signature_decode(m.decode(zz),S)[0])
def run(seed,device,steps=240,nprog=60):
    ms=[];oad=[];ins=[];stats=[]
    for i,(op,arch) in enumerate(zip(OPS,ARCHS)):
      mm,raw=q.train(op,arch,seed*100+i*23+7,device,steps);A,B=q.input_adapter(mm.codec);ad=q.fit_out_identity(mm,op,A,B,device);ms.append(mm);oad.append(ad);ins.append((A,B));exact=sum(q.run_source(mm,ad,A,B,x,y,device)==h.op_apply(op,x,y) for x in range(N) for y in range(N))/(N*N);stats.append({'op':op,'arch':arch,'exact':exact})
    routes,dists,sel=learn_routes(ms,ins,seed,device);S,smeta=observed_S(ms,oad,ins,routes,device);lm,z,lstat=train_latent(S);cache={}
    def prob(i,x,y):
      k=(i,x,y)
      if k not in cache:
        A,B=ins[i];cache[k]=q.action_prob(ms[i],oad[i],A,B,x,y,device)
      return cache[k]
    rng=random.Random(seed+8800);res={};agg={'explicit':0,'latent':0,'programs':0,'agree':0,'stages':0}
    for d in range(1,6):
      ec=lc=agree=stages=0
      for _ in range(nprog):
        ve=vl=t=rng.randrange(N)
        for __ in range(d):
          op=rng.choice(OPS);b=rng.randrange(N);pi,si=routes[op];p1=prob(pi,ve,b);p2=prob(si,ve,b);ye=int((.6*p1+.4*p2).argmax());p1l=prob(pi,vl,b);p2l=prob(si,vl,b);yl=ldecode(p1l,p2l,lm,z,S);agree+=int(ve==vl and ye==yl);stages+=1;ve,vl=ye,yl;t=h.op_apply(op,t,b)
        ec+=ve==t;lc+=vl==t
      res[str(d)]={'explicit':ec/nprog,'behavior_latent5':lc/nprog,'paired_stage_agree':agree/stages};agg['explicit']+=ec;agg['latent']+=lc;agg['programs']+=nprog;agg['agree']+=agree;agg['stages']+=stages
    return {'seed':seed,'source_exact':stats,'routes':{o:{'top':OPS[routes[o][0]],'second':OPS[routes[o][1]],'distances':dists[o]} for o in OPS},'route_exact':sum(routes[o][0]==OPS.index(o) for o in OPS)/6,'signature':smeta,'latent':lstat,'depth':res,'aggregate':agg,'selected_relation_features':sel}
def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--steps',type=int,default=240);ap.add_argument('--out',required=True);x=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);dev=torch.device('cpu');st=time.time();r=run(x.seed,dev,x.steps);out={'experiment':'integrated_auto_relation_identity_output_behavior_latent5_v1','routing':'automatically mined label-invariant generic intervention fingerprint; no expected actions or op-specific contracts','input_alignment':'32 diagonal native-input examples','output_alignment':'right-identity relation only','latent':'5D from equality signatures of observed seen-source rollouts + fusion-behavior objective; heldout MUL excluded','result':r,'elapsed_s':time.time()-st};Path(x.out).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
