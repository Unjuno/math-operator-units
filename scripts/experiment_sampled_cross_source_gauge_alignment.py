from __future__ import annotations
import json,random,statistics,time,math
from pathlib import Path
N=32;OPS=('add','sub','min','max','xor','mul')
def f(op,a,b):
    if op=='add':return (a+b)%N
    if op=='sub':return (a-b)%N
    if op=='min':return min(a,b)
    if op=='max':return max(a,b)
    if op=='xor':return a^b
    if op=='mul':return (a*b)%N
class Oracle:
    def __init__(self,op,p):
        self.op=op;self.p=p;self.inv=[0]*N;self.q=0
        for s,l in enumerate(p):self.inv[l]=s
    def __call__(self,a,b):
        self.q+=1;return self.p[f(self.op,self.inv[a],self.inv[b])]

def merge_sort_labels(labels,cmp):
    if len(labels)<=1:return labels
    m=len(labels)//2;a=merge_sort_labels(labels[:m],cmp);b=merge_sort_labels(labels[m:],cmp);out=[];i=j=0
    while i<len(a) and j<len(b):
        if cmp(a[i],b[j]):out.append(a[i]);i+=1
        else:out.append(b[j]);j+=1
    return out+a[i:]+b[j:]

def find_identity(Q,two_sided=True):
    xs=[0,1]
    for e in range(N):
        ok=True
        for x in xs:
            if Q(x,e)!=x or (two_sided and Q(e,x)!=x):ok=False;break
        if not ok:continue
        if all(Q(x,e)==x and (not two_sided or Q(e,x)==x) for x in range(N)):return e
    raise RuntimeError('identity')
def add_map(Q,bridge):
    e=find_identity(Q,True)
    for g in range(N):
        cur=e;seq=[];seen=set()
        while cur not in seen and len(seq)<=N:
            seq.append(cur);seen.add(cur);cur=Q(cur,g)
        if cur==e and len(seq)==N:
            u=bridge(g);return {l:(k*u)%N for k,l in enumerate(seq)},1
    raise RuntimeError('generator')
def sub_map(Q,bridge):
    e=find_identity(Q,False)
    for g in range(N):
        ng=Q(e,g);cur=e;seq=[];seen=set()
        while cur not in seen and len(seq)<=N:
            seq.append(cur);seen.add(cur);cur=Q(cur,ng)
        if cur==e and len(seq)==N:
            u=bridge(g);return {l:(k*u)%N for k,l in enumerate(seq)},1
    raise RuntimeError('generator')
def xor_map(Q,bridge):
    e=find_identity(Q,True);ctol={0:e};ltoc={e:0};basis=[]
    for cand in range(N):
        if cand in ltoc:continue
        bit=1<<len(basis);basis.append(cand)
        for c,l in list(ctol.items()):nl=Q(l,cand);ctol[c|bit]=nl;ltoc[nl]=c|bit
        if len(basis)==5:break
    cols=[bridge(l) for l in basis];m={}
    for l,c in ltoc.items():
        y=0
        for i,v in enumerate(cols):
            if (c>>i)&1:y^=v
        m[l]=y
    return m,5
def min_map(Q):
    labels=list(range(N));ordered=merge_sort_labels(labels,lambda x,y: Q(x,y)==x)
    return {l:i for i,l in enumerate(ordered)}
def max_map(Q):
    labels=list(range(N));ordered=merge_sort_labels(labels,lambda x,y: Q(x,y)==y)
    return {l:i for i,l in enumerate(ordered)}
def generated_mul(vals):
    S={1}|set(vals);changed=True
    while changed:
        changed=False
        for a in list(S):
            for b in list(S):
                c=(a*b)%N
                if c not in S:S.add(c);changed=True
    return S
def mul_identity(Q):
    ids=[x for x in range(N) if Q(x,x)==x];assert len(ids)==2
    a,b=ids;w=next(x for x in range(N) if x not in ids)
    return a if Q(a,w)==w else b
def mul_map(Q,bridge,rng):
    one=mul_identity(Q);pool=list(range(N));rng.shuffle(pool);gens=[];gvals=[];bq=0
    for l in pool:
        if l==one:continue
        v=bridge(l);bq+=1
        if len(generated_mul(gvals+[v]))>len(generated_mul(gvals)):
            gens.append(l);gvals.append(v)
        if len(generated_mul(gvals))==N:break
    assert len(generated_mul(gvals))==N
    m={one:1}
    for l,v in zip(gens,gvals):
        if l in m:assert m[l]==v
        m[l]=v
    queue=list(m)
    while queue:
        l=queue.pop(0);v=m[l]
        for gl,gv in zip(gens,gvals):
            nl=Q(l,gl);nv=(v*gv)%N
            if nl not in m:m[nl]=nv;queue.append(nl)
            else:assert m[nl]==nv
    assert len(m)==N
    return m,bq

def trial(seed,nprog=60):
    rng=random.Random(seed);ps={};Qs={}
    for op in OPS:
        p=list(range(N));rng.shuffle(p);ps[op]=p;Qs[op]=Oracle(op,p)
    maps={'min':min_map(Qs['min']),'max':max_map(Qs['max'])};bq={'min':0,'max':0}
    invs={op:[0]*N for op in OPS}
    for op in OPS:
        for s,l in enumerate(ps[op]):invs[op][l]=s
    def bridge(op,l):
        hidden=invs[op][l];min_label=ps['min'][hidden];return maps['min'][min_label]
    maps['add'],bq['add']=add_map(Qs['add'],lambda l:bridge('add',l))
    maps['sub'],bq['sub']=sub_map(Qs['sub'],lambda l:bridge('sub',l))
    maps['xor'],bq['xor']=xor_map(Qs['xor'],lambda l:bridge('xor',l))
    maps['mul'],bq['mul']=mul_map(Qs['mul'],lambda l:bridge('mul',l),rng)
    exact={op:sum(maps[op][ps[op][s]]==s for s in range(N))/N for op in OPS}
    calibration_local_queries={op:Qs[op].q for op in OPS}
    invmap={}
    for op,m in maps.items():
        z=[None]*N
        for l,s in m.items():z[s]=l
        assert all(x is not None for x in z);invmap[op]=z
    e2e={}
    for d in range(1,11):
        ok=0
        for _ in range(nprog):
            v=truth=rng.randrange(N)
            for __ in range(d):
                op=rng.choice(OPS);b=rng.randrange(N);lo=Qs[op](invmap[op][v],invmap[op][b]);v=maps[op][lo];truth=f(op,truth,b)
            ok+=v==truth
        e2e[str(d)]=ok/nprog
    localq={op:Qs[op].q for op in OPS}
    return {'bridge_queries':bq,'calibration_local_queries':calibration_local_queries,'local_queries_total_including_eval':localq,'mapping_exact':exact,'e2e':e2e}
def main():
    st=time.time();ts=[trial(50000+i) for i in range(100)]
    bmean={op:statistics.mean(t['bridge_queries'][op] for t in ts) for op in OPS}
    out={'experiment':'sampled_cross_source_gauge_alignment_v1','N':N,'trials':100,
      'method':'no exhaustive private transition tables: MIN/MAX use comparison sorting; ADD/SUB discover identity+cyclic generator; XOR builds a five-vector basis; MUL samples bridge-labeled generators until they generate the monoid, then expands by local multiplication queries',
      'mean_bridge_queries':bmean,'mean_total_bridge_queries':sum(bmean.values()),
      'mul_bridge_query_distribution':{'mean':statistics.mean(t['bridge_queries']['mul'] for t in ts),'median':statistics.median(t['bridge_queries']['mul'] for t in ts),'max':max(t['bridge_queries']['mul'] for t in ts)},
      'mean_calibration_local_queries':{op:statistics.mean(t['calibration_local_queries'][op] for t in ts) for op in OPS},
      'mean_total_calibration_local_queries':statistics.mean(sum(t['calibration_local_queries'].values()) for t in ts),
      'full_table_local_queries_baseline':len(OPS)*N*N,
      'local_query_reduction_vs_full_tables':1-statistics.mean(sum(t['calibration_local_queries'].values()) for t in ts)/(len(OPS)*N*N),
      'all_mappings_exact':all(all(v==1 for v in t['mapping_exact'].values()) for t in ts),'all_e2e_exact':all(all(v==1 for v in t['e2e'].values()) for t in ts),
      'mean_e2e':{str(d):statistics.mean(t['e2e'][str(d)] for t in ts) for d in range(1,11)},
      'claim_boundary':'still assumes factorized private state labels, known operation family, and same-latent-state bridge into MIN; query counts exclude downstream program execution and measure calibration only',
      'elapsed_s':time.time()-st}
    p=Path('/mnt/data/opfusion-local/evaluations/local_causal_saliency/sampled_cross_source_gauge_alignment_v1.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));print('WROTE',p)
if __name__=='__main__':main()
