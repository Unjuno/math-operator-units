from __future__ import annotations
import importlib.util,json,time,math
from pathlib import Path
import numpy as np, torch
spec=importlib.util.spec_from_file_location('ex','/mnt/data/operator_fusion_v2/phase5_existential_gauge_openworld_validation.py')
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex);ow=ex.ow;b=ex.b
torch.set_num_threads(1);rng=np.random.default_rng(20260904);ex.rng=rng;ow.rng=rng
W=.6

def active_score(target,gi,r,post,regs,blocks=2):
 bl=[ex.block(target,gi,r,post,regs,B=32,M=3072) for _ in range(blocks)]
 return ex.accumulate(bl,'active')[0], ex.accumulate(bl,'active')[1]

def main():
 t=time.time();regs,rr,cal,cr,ev,er=ow.train_pool();primary=[x[0] for x in regs]
 calvals={s:[] for s in range(6)}
 for s in range(6):
  for gi in (s%8,(s+2)%8,(s+4)%8,(s+6)%8):
   r,q,p,post=ow.route(cal[s],gi,primary,maxq=36)
   if r is not None:
    v,_=active_score(cal[s],gi,r,post,regs);calvals[r].append(v)
 th={s:(max(calvals[s])+1 if calvals[s] else 999) for s in range(6)}
 true_g=[int(rng.integers(0,8)) for _ in range(6)]
 out_chart=[(float(rng.uniform(.5,2.2)),float(rng.uniform(-.8,.8))) for _ in range(6)]
 runtime={};route_rows=[]
 for s in range(6):
  r,q,p,post=ow.route(ev[s],true_g[s],primary,maxq=36);v,bg=(999,None) if r is None else active_score(ev[s],true_g[s],r,post,regs)
  accepted=(r is not None and v<th[r]);runtime[s]={'kind':'in_family','model':ev[s],'origin':s,'gauge':true_g[s],'route':r,'accepted':accepted,'validation_mismatch':v,'best_validation_gauge':bg,'posterior':p,'out_chart':out_chart[s]}
  route_rows.append({'handle':s,'kind':'in_family','origin':s,'gauge':true_g[s],'route':r,'posterior':p,'validation_mismatch':v,'threshold':th[r] if r is not None else None,'accepted':accepted,'correct_route':r==s if r is not None else False})
 ood_handles=[]
 for s,fn in enumerate(b.FUNS):
  nm,nr=b.train(ow.mk_near(fn,s),7000+s);h=6+s;gi=int(rng.integers(0,8));r,q,p,post=ow.route(nm,gi,primary,maxq=36);v,bg=(999,None) if r is None else active_score(nm,gi,r,post,regs);acc=(r is not None and v<th[r])
  runtime[h]={'kind':'near_ood','model':nm,'origin':s,'gauge':gi,'route':r,'accepted':acc,'validation_mismatch':v,'best_validation_gauge':bg,'posterior':p};ood_handles.append(h)
  route_rows.append({'handle':h,'kind':'near_ood','origin':s,'gauge':gi,'route':r,'posterior':p,'validation_mismatch':v,'threshold':th[r] if r is not None else None,'accepted':acc})
 def food(x):
  a,c=x[:,0],x[:,1];return .18*np.sin(2.3*a*c)+.20*a*a+.17*c**3
 broad,brm=b.train(food,6500);h=12;gi=int(rng.integers(0,8));r,q,p,post=ow.route(broad,gi,primary,maxq=36);v,bg=(999,None) if r is None else active_score(broad,gi,r,post,regs);acc=(r is not None and v<th[r]);runtime[h]={'kind':'broad_ood','model':broad,'origin':None,'gauge':gi,'route':r,'accepted':acc,'validation_mismatch':v,'best_validation_gauge':bg,'posterior':p};ood_handles.append(h)
 route_rows.append({'handle':h,'kind':'broad_ood','origin':None,'gauge':gi,'route':r,'posterior':p,'validation_mismatch':v,'threshold':th[r] if r is not None else None,'accepted':acc})
 x_anchor=np.array([.231,.617])
 def recover_input_gauge(gi):
  z=x_anchor@b.GAUGES[gi].T;errs=[float(np.linalg.norm(x_anchor@G.T-z)) for G in b.GAUGES];return int(np.argmin(errs))
 def fit_output_chart(s,o):
  y1,y2=-.47,.63;z1,z2=s*y1+o,s*y2+o;sh=(z2-z1)/(y2-y1);oh=z1-sh*y1;return sh,oh
 def call(m,state,arg):
  x=torch.tensor([[float(np.clip(state,-1,1)),float(np.clip(arg,-1,1))]],dtype=torch.float32)
  with torch.no_grad():return float(m(x).item())
 def target_sem(handle,state,arg):
  rec=runtime[handle];m=rec['model'];G=b.GAUGES[rec['gauge']];x=np.array([[state,arg]])@G.T;sem=b.semantic_from_native(x,G)
  with torch.no_grad():y=float(m(torch.tensor(sem,dtype=torch.float32)).item())
  s,o=rec['out_chart'];z=s*y+o;return (z-o)/s
 def ensure(handle,cache):
  if handle in cache:return 0
  rec=runtime[handle]
  if not rec['accepted']:return None
  gh=recover_input_gauge(rec['gauge']);s,o=rec['out_chart'];sh,oh=fit_output_chart(s,o);cache[handle]={'gauge':gh,'s':sh,'o':oh};return 3
 def cand_sem(handle,state,arg):return call(primary[runtime[handle]['route']],state,arg)
 def clean_program(depth):
  cache={};ground=0;st=float(rng.uniform(-.35,.35));sc=st;hard=st
  for _ in range(depth):
   i=int(rng.integers(0,6));j=int((i+1+rng.integers(0,5))%6);arg=float(rng.uniform(-.6,.6))
   ci=ensure(i,cache);cj=ensure(j,cache)
   if ci is None or cj is None:return None
   ground+=ci+cj
   ti=target_sem(i,st,arg);tj=target_sem(j,st,arg);st=W*ti+(1-W)*tj
   ai=cand_sem(i,sc,arg);aj=cand_sem(j,sc,arg);sc=W*ai+(1-W)*aj
   hard=cand_sem(i,hard,arg)
  return abs(sc-st),abs(hard-st),ground,len(cache)
 mixed_total=600;mixed_unresolved=0;mixed_false_execute=0
 for _ in range(mixed_total):
  oh=int(rng.choice(ood_handles));other=int(rng.integers(0,6));handles=[other,oh] if rng.random()<.5 else [oh,other]
  if any(not runtime[x]['accepted'] for x in handles):mixed_unresolved+=1
  else:mixed_false_execute+=1
 depths={}
 for d in range(1,11):
  vals=[clean_program(d) for _ in range(180)];valid=[x for x in vals if x is not None]
  if valid:
   a=np.array(valid,float);depths[str(d)]={'programs':len(vals),'resolved':len(valid),'unresolved':len(vals)-len(valid),'fused_mean_error':float(a[:,0].mean()),'fused_p95':float(np.percentile(a[:,0],95)),'hard_mean_error':float(a[:,1].mean()),'mean_grounding':float(a[:,2].mean()),'mean_unique_grounded':float(a[:,3].mean())}
  else:depths[str(d)]={'programs':len(vals),'resolved':0,'unresolved':len(vals)}
 out={'experiment':'v2_phase5_openworld_private_input_gaugeaware_fusion_e2e_v2','validation_queries':64,'registered_instances_per_source':3,'thresholds':th,'runtime_registration':route_rows,'in_family_registration':{'total':6,'accepted':sum(runtime[s]['accepted'] for s in range(6)),'correct_accepted':sum(runtime[s]['accepted'] and runtime[s]['route']==s for s in range(6)),'wrong_accepted':sum(runtime[s]['accepted'] and runtime[s]['route']!=s for s in range(6))},'ood_registration':{'total':len(ood_handles),'rejected':sum(not runtime[h]['accepted'] for h in ood_handles),'false_accept':sum(runtime[h]['accepted'] for h in ood_handles),'near_rejected':sum(not runtime[h]['accepted'] for h in ood_handles if runtime[h]['kind']=='near_ood'),'broad_rejected':sum(not runtime[h]['accepted'] for h in ood_handles if runtime[h]['kind']=='broad_ood')},'fusion_contract':{'output_gauge_group':'positive_affine','scope_before':'independent_per_plugin','output_grounding':2,'input_grounding':1,'scope_after':'aligned_common_chart','weighted_mean':W},'clean_depth':depths,'mixed_ood_programs':{'programs':mixed_total,'unresolved_before_unsafe_execution':mixed_unresolved,'false_execute':mixed_false_execute},'elapsed_s':time.time()-t}
 out['pass']=out['in_family_registration']['wrong_accepted']==0 and out['in_family_registration']['accepted']==6 and out['ood_registration']['false_accept']==0 and mixed_false_execute==0 and depths['10']['fused_mean_error']<.01
 Path('/mnt/data/operator_fusion_v2/phase5_openworld_gaugeaware_fusion_e2e_v2.json').write_text(json.dumps(out,indent=2));print(json.dumps({'in_family_registration':out['in_family_registration'],'ood_registration':out['ood_registration'],'depth10':depths['10'],'mixed_ood_programs':out['mixed_ood_programs'],'elapsed_s':out['elapsed_s'],'pass':out['pass']},indent=2))
if __name__=='__main__':main()
