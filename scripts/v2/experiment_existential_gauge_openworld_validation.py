from __future__ import annotations
import importlib.util,json,time,math
from pathlib import Path
import numpy as np, torch
spec=importlib.util.spec_from_file_location('ow','/mnt/data/operator_fusion_v2/phase5_private_input_openworld_routing_validation.py')
ow=importlib.util.module_from_spec(spec);spec.loader.exec_module(ow);b=ow.b
torch.set_num_threads(1);rng=np.random.default_rng(20260903);ow.rng=rng

def outputs(m,z,G):
 x=torch.tensor(b.semantic_from_native(z,G),dtype=torch.float32)
 with torch.no_grad():return m(x).numpy()

def plausible_gauges(selected,post,mass=.995,min_prob=.002):
 gp=post[selected*8:(selected+1)*8].copy();gp/=gp.sum();order=np.argsort(gp)[::-1];out=[];cum=0
 for gi in order:
  if gp[gi]<min_prob and cum>=mass:break
  out.append(int(gi));cum+=float(gp[gi])
  if cum>=mass:break
 return out,gp

def block(target,true_gauge,selected,post,regs,B=32,M=3072):
 gs,gp=plausible_gauges(selected,post)
 z1=rng.uniform(-1,1,(M,2));z2=rng.uniform(-1,1,(M,2))
 gvote={};gmargin={};gcons={}
 for gi in gs:
  bits=[];marg=[]
  for m in regs[selected]:
   y1=outputs(m,z1,b.GAUGES[gi]);y2=outputs(m,z2,b.GAUGES[gi]);bits.append(y1<y2);marg.append(np.abs(y1-y2))
  bits=np.stack(bits);marg=np.stack(marg);gvote[gi]=bits.mean(0)>=.5;gmargin[gi]=np.median(marg,axis=0);gcons[gi]=np.all(bits==bits[:1],axis=0)
 picks=[];quota=max(1,int(math.ceil(B/len(gs))))
 for gi in gs:
  idx=np.where(gcons[gi])[0]
  if len(idx)==0:idx=np.arange(M)
  vals=gmargin[gi][idx];qlo=np.quantile(vals,.005) if len(idx)>20 else -1.;idx2=idx[vals>=qlo]
  order=idx2[np.argsort(gmargin[gi][idx2])]
  for j in order[:quota*3]:
   if int(j) not in picks:picks.append(int(j))
   if len(picks)>=B:break
  if len(picks)>=B:break
 if len(picks)<B:
  score=np.min(np.stack([gmargin[g] for g in gs]),axis=0)
  for j in np.argsort(score):
   if int(j) not in picks:picks.append(int(j))
   if len(picks)>=B:break
 aidx=np.array(picks[:B],int);ridx=rng.choice(M,B,replace=False)
 obs=b.eval_rel(target,z1,z2,b.GAUGES[true_gauge])
 return {'gauges':gs,
         'active_by_gauge':{str(g):int(np.sum(gvote[g][aidx]!=obs[aidx])) for g in gs},
         'random_by_gauge':{str(g):int(np.sum(gvote[g][ridx]!=obs[ridx])) for g in gs}}

def accumulate(blocks,mode):
 if not blocks:return 999,None
 gs=blocks[0]['gauges'];sums={g:0 for g in gs}
 for bl in blocks:
  for g in gs:sums[g]+=bl[mode+'_by_gauge'][str(g)]
 best=min(sums,key=sums.get);return int(sums[best]),int(best)

def main():
 t=time.time();regs,rr,cal,cr,ev,er=ow.train_pool();primary=[x[0] for x in regs];KMAX=3
 caldata={s:{'random':[[] for _ in range(KMAX)],'active':[[] for _ in range(KMAX)]} for s in range(6)};calroutes=[]
 for s in range(6):
  for gi in ((s)%8,(s+2)%8,(s+4)%8,(s+6)%8):
   r,q,p,post=ow.route(cal[s],gi,primary)
   if r is None:continue
   bl=[block(cal[s],gi,r,post,regs) for _ in range(KMAX)]
   rec={'source':s,'gauge':gi,'route':r,'queries':q,'posterior':p,'gauges':bl[0]['gauges']}
   for k in range(KMAX):
    for mode in ('random','active'):
     v,bg=accumulate(bl[:k+1],mode);caldata[r][mode][k].append(v);rec[f'{mode}_{32*(k+1)}']={'min_mismatch':v,'best_gauge':bg}
   calroutes.append(rec)
 thresholds={s:{m:[max(caldata[s][m][k])+1 if caldata[s][m][k] else 999 for k in range(KMAX)] for m in ('random','active')} for s in range(6)}
 cases=[]
 for s in range(6):
  for gi in ((2*s+1)%8,(2*s+5)%8):cases.append(('in_family',s,ev[s],gi))
 def food(x):
  a,c=x[:,0],x[:,1];return .18*np.sin(2.3*a*c)+.20*a*a+.17*c**3
 broad,brm=b.train(food,6500)
 for gi in range(8):cases.append(('broad_ood',None,broad,gi))
 near_rm=[]
 for s,fn in enumerate(b.FUNS):
  nm,nr=b.train(ow.mk_near(fn,s),7000+s);near_rm.append(nr)
  for gi in ((s+1)%8,(s+4)%8):cases.append(('near_ood',s,nm,gi))
 rows=[]
 for kind,origin,target,gi in cases:
  r,q,p,post=ow.route(target,gi,primary);row={'kind':kind,'origin_source':origin,'gauge':gi,'route':r,'queries':q,'posterior':p}
  if r is not None:
   bl=[block(target,gi,r,post,regs) for _ in range(KMAX)];row['plausible_gauges']=bl[0]['gauges']
   for k in range(KMAX):
    for mode in ('random','active'):
     v,bg=accumulate(bl[:k+1],mode);row[f'{mode}_{32*(k+1)}']={'min_mismatch':v,'best_gauge':bg}
  rows.append(row)
 summaries={}
 for k in range(KMAX):
  Q=32*(k+1);summaries[str(Q)]={}
  for mode in ('random','active'):
   def reject(x):
    if x['route'] is None:return True
    return x[f'{mode}_{Q}']['min_mismatch']>=thresholds[x['route']][mode][k]
   inf=[x for x in rows if x['kind']=='in_family'];ood=[x for x in rows if x['kind']!='in_family'];acc=[x for x in inf if not reject(x)]
   summaries[str(Q)][mode]={'in_total':len(inf),'in_accept':len(acc),'in_correct_accept':sum(x['route']==x['origin_source'] for x in acc),'in_wrong_accept':sum(x['route']!=x['origin_source'] for x in acc),'in_unresolved':sum(reject(x) for x in inf),'ood_total':len(ood),'ood_reject':sum(reject(x) for x in ood),'ood_false_accept':sum(not reject(x) for x in ood),'broad_reject':sum(reject(x) for x in ood if x['kind']=='broad_ood'),'near_reject':sum(reject(x) for x in ood if x['kind']=='near_ood')}
 out={'experiment':'v2_phase5_existential_latent_gauge_openworld_validation','registered_instances':3,'query_block':32,'blocks':KMAX,'proposal_per_block':3072,'thresholds':thresholds,'calibration_routes':calroutes,'summaries':summaries,'rows':rows,'near_rmse':near_rm,'broad_rmse':brm,'design':'input gauge remains a latent nuisance; source is rejected only if every plausible gauge hypothesis fails independent heldout behavioral stress tests','runtime_constraint':'target-native coordinates only; no common semantic coordinate is exposed','elapsed_s':time.time()-t}
 out['pass']=all(summaries[q]['active']['in_wrong_accept']==0 for q in summaries) and summaries['96']['active']['ood_false_accept']<=2
 Path('/mnt/data/operator_fusion_v2/phase5_existential_gauge_openworld_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps({'summaries':summaries,'thresholds':thresholds,'elapsed_s':out['elapsed_s'],'pass':out['pass']},indent=2))
if __name__=='__main__':main()
