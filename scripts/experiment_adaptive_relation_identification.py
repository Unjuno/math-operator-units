from __future__ import annotations
import json,statistics,time
from pathlib import Path
import numpy as np, sys
sys.path.insert(0,'/mnt/data/opfusion-local/work')
import experiment_random_intervention_discovery as r
N=r.N;OPS=r.OPS;NAMES=list(OPS)+['shadow']

def intfeat(name,idx,item):
    x,y,_=item;base=r.BASE[name][idx];vals=(r.shadow(x,y) if name=='shadow' else r.f(name,x,y))[idx];e,mx,u=r.ent(vals,len(idx));return np.array([float(np.mean(base==vals)),u,e,mx])
def trial(seed,pool=128,n=96,maxq=8,topc=3):
    lib=r.library(seed,pool);rng=np.random.default_rng(seed+778);taskidx=rng.choice(N*N,n,False);full=np.arange(N*N);src={s:np.stack([intfeat(s,full,item) for item in lib]) for s in NAMES};results={}
    for target in OPS:
        task=np.stack([intfeat(target,taskidx,item) for item in lib]);cum={s:0.0 for s in NAMES};used=[];traj=[];candidates=NAMES[:]
        for q in range(1,maxq+1):
            bestj=None;best=-1
            for j in range(pool):
                if j in used:continue
                mat=np.stack([src[s][j] for s in candidates]);score=float(mat.var(0).sum())
                if score>best:best=score;bestj=j
            used.append(bestj);obs=task[bestj]
            for s in NAMES:cum[s]+=float(np.mean((obs-src[s][bestj])**2))
            rank=sorted((cum[s]/q,s) for s in NAMES);candidates=[s for _,s in rank[:topc]];traj.append({'q':q,'top':rank[0][1],'second':rank[1][1],'gap':rank[1][0]-rank[0][0],'chosen':lib[bestj][2]})
        results[target]=traj
    return results
def main():
    st=time.time();tr=[trial(400000+i) for i in range(100)];rows=[]
    for q in range(1,9):
        c=tot=add=shadow=0;g=[]
        for t in tr:
            for op in OPS:
                z=t[op][q-1];c+=z['top']==op;tot+=1;g.append(z['gap'])
                if op=='add':add+=z['top']=='add';shadow+=z['top']=='shadow'
        rows.append({'queries':q,'route_acc':c/tot,'all6_trial_rate':sum(all(t[o][q-1]['top']==o for o in OPS) for t in tr)/len(tr),'true_add_rate':add/len(tr),'shadow_top_rate':shadow/len(tr),'mean_gap':statistics.mean(g)})
    abst=[]
    for q in [2,3,4,5,6,8]:
        dev=[]
        for t in tr[:50]:
            for op in OPS[:-1]:
                z=t[op][q-1];dev.append((z['gap'],z['top']==op))
        ths=sorted(set(x for x,_ in dev));bestth=0;bestscore=-1
        for th in ths:
            resolved=[(g,c) for g,c in dev if g>=th];wrong=sum(not c for _,c in resolved);score=sum(c for _,c in resolved)-5*wrong
            if score>bestscore:bestscore=score;bestth=th
        right=wrong=unres=0
        for t in tr[50:]:
            for op in OPS:
                z=t[op][q-1]
                if z['gap']<bestth:unres+=1
                elif z['top']==op:right+=1
                else:wrong+=1
        abst.append({'queries':q,'dev_gap_threshold':bestth,'resolved_correct':right,'resolved_wrong':wrong,'unresolved':unres,'resolution_rate':right/(right+wrong+unres)})
    out={'experiment':'adaptive_label_invariant_relation_identification_v1','candidate_interventions':128,'task_samples_per_query':96,'candidate_sources':'six legitimate sources plus SHADOW nuisance','method':'after each label-invariant task observation, choose next random-DSL intervention maximizing variance among current top-3 candidate source fingerprints; no expected action labels or op-specific contracts','forced_rows':rows,'calibrated_abstention':abst,'claim_boundary':'source fingerprints use exhaustive symbolic source behavior here; generic DSL and common input intervention API remain provided','elapsed_s':time.time()-st}
    p=Path('/mnt/data/opfusion-local/evaluations/local_causal_saliency/adaptive_relation_identification_v1.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
