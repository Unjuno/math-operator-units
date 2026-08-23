from __future__ import annotations
import json, numpy as np, torch
from pathlib import Path
import experiment_neural_private_input_routing as b

rng=np.random.default_rng(20260824); torch.set_num_threads(1)
cands=[]; cr=[]
for sd in [3002,3102,3202]:
    m,r=b.train(b.f2,sd); cands.append(m); cr.append(r)
in_target,ir=b.train(b.f2,4002)
delta=.02
def near_fn(x):
    a,c=x[:,0],x[:,1]
    return b.f2(x)+delta*np.sin(4.3*a+1.7*c)
near,nr=b.train(near_fn,7020)

def outputs(model,z,G):
    x=torch.tensor(b.semantic_from_native(z,G),dtype=torch.float32)
    with torch.no_grad(): return model(x).numpy()

def one(target,G,B,M=4096):
    z1=rng.uniform(-1,1,(M,2)); z2=rng.uniform(-1,1,(M,2)); bits=[]; margins=[]
    for m in cands:
        y1=outputs(m,z1,G); y2=outputs(m,z2,G); bits.append(y1<y2); margins.append(np.abs(y1-y2))
    bits=np.stack(bits); margins=np.stack(margins); vote=(bits.mean(0)>=.5); consensus=np.all(bits==bits[0:1],axis=0); med=np.median(margins,axis=0)
    y1=outputs(target,z1,G); y2=outputs(target,z2,G); obs=y1<y2
    ridx=rng.choice(M,B,replace=False); cidx=np.where(consensus)[0]
    tidx=cidx[np.argsort(med[cidx])[:B]] if len(cidx)>=B else np.argsort(med)[:B]
    qlo=np.quantile(med[cidx],.005) if len(cidx) else 0; cidx2=cidx[med[cidx]>=qlo]
    sidx=cidx2[np.argsort(med[cidx2])[:B]] if len(cidx2)>=B else tidx
    return int(np.sum(vote[ridx]!=obs[ridx])),int(np.sum(vote[tidx]!=obs[tidx])),int(np.sum(vote[sidx]!=obs[sidx]))

def main():
    rows={}
    for B in [16,32,64]:
        dat={'in_r':[],'in_t':[],'in_s':[],'near_r':[],'near_t':[],'near_s':[]}
        for G in b.GAUGES:
            for _ in range(30):
                a,c,d=one(in_target,G,B); dat['in_r'].append(a); dat['in_t'].append(c); dat['in_s'].append(d)
                a,c,d=one(near,G,B); dat['near_r'].append(a); dat['near_t'].append(c); dat['near_s'].append(d)
        rr={}
        for key,label in [('r','random'),('t','consensus_low_margin'),('s','consensus_low_margin_trimmed')]:
            iv=np.array(dat['in_'+key]); nv=np.array(dat['near_'+key]); thr=int(iv.max()+1)
            rr[label]={'in_mean':float(iv.mean()),'in_max':int(iv.max()),'zero_observed_false_reject_threshold':thr,'near_mean':float(nv.mean()),'near_detected':int(np.sum(nv>=thr)),'samples':len(iv)}
        rows[str(B)]=rr
    out={'experiment':'v2_phase5_neural_multiinstance_stress_validation','registered_instances':3,'candidate_rmse':cr,'heldout_in_family_rmse':ir,'near_ood_rmse':nr,'near_delta':delta,'results':rows,
      'note':'query selection uses only registered-instance agreement and native-output margins; thresholds calibrated to zero observed false rejection on heldout in-family instance'}
    Path('phase5_neural_multiinstance_stress_validation.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
