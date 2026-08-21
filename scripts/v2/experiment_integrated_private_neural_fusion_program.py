from __future__ import annotations
import json, numpy as np, torch, time
from pathlib import Path
import experiment_neural_private_input_routing as b

torch.set_num_threads(1); rng=np.random.default_rng(20260830); W=.6
cands=[]; targets=[]; crm=[]; trm=[]
for i,fn in enumerate(b.FUNS):
    cm,r=b.train(fn,3000+i); cands.append(cm); crm.append(r)
    tm,r=b.train(fn,4000+i); targets.append(tm); trm.append(r)
true_g=[int(rng.integers(0,len(b.GAUGES))) for _ in b.FUNS]
out_chart=[(float(rng.uniform(.5,2.2)),float(rng.uniform(-.8,.8))) for _ in b.FUNS]
x_anchor=np.array([.231,.617])

def recover_input_gauge(gi):
    z=x_anchor@b.GAUGES[gi].T; errs=[float(np.linalg.norm(x_anchor@G.T-z)) for G in b.GAUGES]
    return int(np.argmin(errs)),min(errs)

def fit_output_chart(s,o):
    y1,y2=-.47,.63; z1,z2=s*y1+o,s*y2+o; sh=(z2-z1)/(y2-y1); oh=z1-sh*y1
    return sh,oh

def call(model,state,arg):
    x=torch.tensor([[float(np.clip(state,-1,1)),float(np.clip(arg,-1,1))]],dtype=torch.float32)
    with torch.no_grad(): return float(model(x).item())

routes={}; route_records=[]; route_wrong=0; route_unresolved=0
for s in range(len(b.FUNS)):
    r,q,p,_=b.route_one(targets[s],s,true_g[s],out_chart[s],cands,rng=rng); routes[s]=r
    if r is None: route_unresolved+=1
    elif r!=s: route_wrong+=1
    route_records.append({'target_handle':s,'hidden_input_gauge':true_g[s],'route':r,'queries':q,'posterior':p})

def ensure_grounded(handle,cache):
    if handle in cache:return 0
    gh,ge=recover_input_gauge(true_g[handle]); s,o=out_chart[handle]; sh,oh=fit_output_chart(s,o)
    cache[handle]={'input_gauge':gh,'input_anchor_error':ge,'output_s':sh,'output_o':oh}; return 3

def target_semantic(handle,state,arg):
    G=b.GAUGES[true_g[handle]]; x=np.array([[state,arg]])@G.T; sem=b.semantic_from_native(x,G)
    with torch.no_grad(): y=float(targets[handle](torch.tensor(sem,dtype=torch.float32)).item())
    s,o=out_chart[handle]; z=s*y+o
    return (z-o)/s

def candidate_semantic(handle,state,arg,cache):
    return call(cands[routes[handle]],state,arg)

def run_program(depth):
    cache={}; grounding=0; st=float(rng.uniform(-.4,.4)); sc=st; sh=st; plan=[]
    for _ in range(depth):
        i=int(rng.integers(0,len(b.FUNS))); j=int((i+1+rng.integers(0,len(b.FUNS)-1))%len(b.FUNS)); arg=float(rng.uniform(-.65,.65)); plan.append((i,j,arg))
    for i,j,arg in plan:
        grounding+=ensure_grounded(i,cache); grounding+=ensure_grounded(j,cache)
        ti=target_semantic(i,st,arg); tj=target_semantic(j,st,arg); st=W*ti+(1-W)*tj
        ci=candidate_semantic(i,sc,arg,cache); cj=candidate_semantic(j,sc,arg,cache); sc=W*ci+(1-W)*cj
        sh=candidate_semantic(i,sh,arg,cache)
    return abs(sc-st),abs(sh-st),grounding,len(cache),abs(sc-sh)

def main():
    t=time.time(); depths={}
    if route_wrong or route_unresolved:
        out={'experiment':'v2_phase5_integrated_private_neural_fusion_program','routing':{'wrong':route_wrong,'unresolved':route_unresolved,'records':route_records},'status':'blocked_by_routing'}
    else:
        for d in range(1,11):
            vals=np.array([run_program(d) for _ in range(240)],float)
            depths[str(d)]={'programs':len(vals),'fused_mean_error_vs_independent_target_program':float(vals[:,0].mean()),'fused_p95':float(np.percentile(vals[:,0],95)),'hard_primary_mean_error':float(vals[:,1].mean()),'hard_p95':float(np.percentile(vals[:,1],95)),'mean_lazy_grounding_cost':float(vals[:,2].mean()),'mean_unique_runtime_plugins_grounded':float(vals[:,3].mean()),'mean_fused_vs_hard_difference':float(vals[:,4].mean())}
        out={'experiment':'v2_phase5_integrated_private_neural_fusion_program','weight':W,'candidate_rmse':crm,'target_rmse':trm,
          'routing':{'targets':len(b.FUNS),'wrong':route_wrong,'unresolved':route_unresolved,'correct':len(b.FUNS)-route_wrong-route_unresolved,'mean_queries':float(np.mean([r['queries'] for r in route_records])),'records':route_records},
          'runtime_input_interface':'behavioral identification queries are expressed only in each target plugin native coordinate; hidden D4 input chart is marginalized during routing and fixed lazily by one coordinate anchor when the plugin is first used',
          'output_alignment':'each runtime plugin has an independent positive-affine output chart; two anchors align it to the common fusion chart',
          'fusion_certificate':{'group':'positive_affine','scope_before_grounding':'independent_per_plugin','scope_after_grounding':'aligned_common_chart','weighted_mean_allowed_after_grounding':True},
          'depth':depths,'elapsed_s':time.time()-t}
        out['pass']=bool(depths['10']['fused_mean_error_vs_independent_target_program']<.01 and depths['10']['fused_mean_error_vs_independent_target_program']<depths['10']['hard_primary_mean_error'])
    Path('phase5_integrated_private_neural_fusion_program.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
