from __future__ import annotations
import json,statistics,time
from pathlib import Path
import numpy as np,sys
sys.path.insert(0,'/mnt/data/opfusion-local/work')
import experiment_auto_contract_mining as a
N=a.N;A=a.A;B=a.B;TS=a.TS
SEEN=('add','sub','min','max','xor')
NOVEL=('mul','and','or','nand','absdiff','avg','satadd','left','right','eq','lt','gcd','mod','rotmix');ALL=SEEN+NOVEL

def fv(name,x,y):
    if name in a.OPS:return a.f_vec(name,x,y)
    if name=='and':return np.bitwise_and(x,y)
    if name=='or':return np.bitwise_or(x,y)
    if name=='nand':return np.bitwise_xor(np.bitwise_and(x,y),N-1)
    if name=='absdiff':return np.abs(x-y)
    if name=='avg':return (x+y)//2
    if name=='satadd':return np.minimum(N-1,x+y)
    if name=='left':return x
    if name=='right':return y
    if name=='eq':return (x==y).astype(int)
    if name=='lt':return (x<y).astype(int)
    if name=='gcd':return np.gcd(x,y)
    if name=='mod':return x%(y+1)
    if name=='rotmix':return np.bitwise_xor(((x<<1)|(x>>4))&(N-1),y)
    raise KeyError(name)
BASE={n:fv(n,A,B) for n in ALL};TRANS={n:[fv(n,x,y) for _,x,y in TS] for n in ALL}
def fp(name,idx):
    base=BASE[name][idx];z=[]
    for v in TRANS[name]:
        ys=v[idx];same=float(np.mean(base==ys));cnt=np.bincount(ys,minlength=N);nz=cnt[cnt>0];nn=len(idx);p=nz/nn;z += [same,len(nz)/nn,float(-(p*np.log(p)).sum()/np.log(max(2,nn))),float(cnt.max()/nn)]
    return np.array(z)
def trial(seed,k=32,n=96):
    rng=np.random.default_rng(seed);si=rng.choice(N*N,n,False);ti=rng.choice(N*N,n,False);src={x:fp(x,si) for x in ALL};mat=np.stack([src[x] for x in SEEN]);sel=np.argsort(mat.var(0))[-k:];correct=0;nov=0;routes={};marg=[]
    for x in ALL:
        t=fp(x,ti);ds=sorted((float(np.mean((t[sel]-src[s][sel])**2)),s) for s in ALL);routes[x]=ds[0][1];correct+=ds[0][1]==x;nov+=x in NOVEL and ds[0][1]==x;marg.append(ds[1][0]-ds[0][0])
    return correct/len(ALL),nov/len(NOVEL),routes,min(marg)
def main():
    st=time.time();rows=[]
    for k in [16,32,64,128,260]:
        rs=[trial(800000+i,k) for i in range(100)];conf={x:{} for x in NOVEL}
        for _,_,rt,_ in rs:
            for x in NOVEL:conf[x][rt[x]]=conf[x].get(rt[x],0)+1
        rows.append({'features_selected_from_seen5':k,'mean_route_acc_all19':statistics.mean(x[0] for x in rs),'mean_route_acc_novel14':statistics.mean(x[1] for x in rs),'all19_rate':sum(x[0]==1 for x in rs)/len(rs),'novel_confusions':conf,'mean_min_margin':statistics.mean(x[3] for x in rs)})
    out={'experiment':'many_heldout_relation_generalization_v1','seen_feature_selection':SEEN,'heldout_operations':NOVEL,'candidate_pool_size':len(ALL),'source_and_task_samples':96,'rows':rows,'claim_boundary':'all operations share the same scalar pair input domain and output alphabet size; this tests relational feature transfer across function families, not arbitrary modalities','elapsed_s':time.time()-st};Path('/mnt/data/opfusion-local/evaluations/local_causal_saliency/many_heldout_relation_generalization_v1.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
