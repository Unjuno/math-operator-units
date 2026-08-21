from __future__ import annotations
import json, numpy as np, torch
from pathlib import Path
import experiment_neural_private_input_routing as b

rng=np.random.default_rng(20260825); torch.set_num_threads(1)
cands=[]; targets=[]; crm=[]; trm=[]
for i,fn in enumerate(b.FUNS):
    cm,r=b.train(fn,3000+i); cands.append(cm); crm.append(r)
    tm,r=b.train(fn,4000+i); targets.append(tm); trm.append(r)
outcharts=[(float(rng.uniform(.5,2.2)),float(rng.uniform(-.8,.8))) for _ in b.FUNS]
x_anchor=np.array([.231,.617])

def recover_gauge(z_anchor):
    errs=[float(np.linalg.norm(x_anchor@G.T-z_anchor)) for G in b.GAUGES]
    return int(np.argmin(errs)),min(errs)

def fit_out_chart(s,o):
    y1,y2=-.47,.63; z1,z2=s*y1+o,s*y2+o
    sh=(z2-z1)/(y2-y1); oh=z1-sh*y1
    return sh,oh

def main():
    records=[]; route_wrong=0; abstain=0; gauge_wrong=0; semerrs=[]; nativeerrs=[]; grounding=[]
    for src in range(len(b.FUNS)):
      for gi,G in enumerate(b.GAUGES):
        route,q,p,_=b.route_one(targets[src],src,gi,outcharts[src],cands,rng=rng)
        if route is None:
            abstain+=1; records.append({'src':src,'gauge':gi,'route':None,'queries':q}); continue
        if route!=src: route_wrong+=1
        z_anchor=x_anchor@G.T; gh,ge=recover_gauge(z_anchor); gauge_wrong += gh!=gi
        s,o=outcharts[src]; sh,oh=fit_out_chart(s,o)
        X=rng.uniform(-1,1,(128,2)); Z=X@G.T; Xhat=b.semantic_from_native(Z,b.GAUGES[gh])
        with torch.no_grad():
            yc=cands[route](torch.tensor(Xhat,dtype=torch.float32)).numpy(); yt=targets[src](torch.tensor(X,dtype=torch.float32)).numpy()
        zc=sh*yc+oh; zt=s*yt+o; se=np.abs(yc-yt); ne=np.abs(zc-zt)
        semerrs.extend(se.tolist()); nativeerrs.extend(ne.tolist()); grounding.append(3)
        records.append({'src':src,'gauge':gi,'route':route,'queries':q,'route_posterior':p,'recovered_gauge':gh,'gauge_error':ge,'mean_semantic_instance_gap':float(se.mean()),'mean_target_native_output_error':float(ne.mean()),'grounding':3})
    out={'experiment':'v2_phase5_neural_private_input_route_ground_execute','candidate_rmse':crm,'target_rmse':trm,'tasks':48,
      'routing':{'wrong':route_wrong,'unresolved':abstain,'correct':48-route_wrong-abstain},
      'input_gauge':{'family':'D4','grounding_per_resolved_target':1,'wrong_after_grounding':gauge_wrong},
      'output_gauge':{'family':'independent_positive_affine','grounding_per_resolved_target':2},
      'execution':{'samples':len(semerrs),'mean_semantic_error_vs_independent_target':float(np.mean(semerrs)),'p95_semantic_error':float(np.percentile(semerrs,95)),'mean_target_native_output_error':float(np.mean(nativeerrs)),'p95_native_error':float(np.percentile(nativeerrs,95))},
      'mean_runtime_grounding_per_resolved_target':float(np.mean(grounding)) if grounding else None,
      'boundary':'finite D4 input gauge and two-point positive-affine output grounding; registered candidate charts are assumed known from registration','records':records}
    out['pass']=bool(route_wrong==0 and abstain==0 and gauge_wrong==0 and np.mean(semerrs)<.02)
    Path('phase5_neural_private_input_e2e.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
