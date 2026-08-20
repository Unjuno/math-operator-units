from __future__ import annotations
import json,random,statistics,time,collections,itertools
from pathlib import Path
N=32; OPS=('add','sub','min','max','xor','mul')
def f(op,a,b):
    if op=='add':return (a+b)%N
    if op=='sub':return (a-b)%N
    if op=='min':return min(a,b)
    if op=='max':return max(a,b)
    if op=='xor':return a^b
    if op=='mul':return (a*b)%N

def mk(op,p):
    inv=[0]*N
    for s,l in enumerate(p):inv[l]=s
    T=[[p[f(op,inv[a],inv[b])] for b in range(N)] for a in range(N)]
    return T,inv

def min_rank(T):return {x:sum(T[x][y]==y for y in range(N))-1 for x in range(N)}
def max_rank(T):return {x:sum(T[x][y]==x for y in range(N))-1 for x in range(N)}
def identity(T,two=True):
    for e in range(N):
        if all(T[x][e]==x for x in range(N)) and (not two or all(T[e][x]==x for x in range(N))):return e

def add_coordinates(T):
    e=identity(T,True)
    for g in range(N):
        cur=e;seq=[];seen=set()
        for _ in range(N):seq.append(cur);seen.add(cur);cur=T[cur][g]
        if cur==e and len(seen)==N:return g,{l:k for k,l in enumerate(seq)}
    raise RuntimeError
def sub_coordinates(T):
    e=identity(T,False);neg={y:T[e][y] for y in range(N)};A=[[T[x][neg[y]] for y in range(N)] for x in range(N)];return add_coordinates(A)
def xor_coordinates(T):
    e=identity(T,True);ctol={0:e};ltoc={e:0};basis=[]
    for cand in range(N):
        if cand in ltoc:continue
        bit=1<<len(basis);basis.append(cand)
        for c,l in list(ctol.items()):nl=T[l][cand];ctol[c|bit]=nl;ltoc[nl]=c|bit
        if len(basis)==5:break
    assert len(ltoc)==N;return basis,ltoc

def power_pattern(T,one,x):
    cur=one;mp={};pat=[]
    for _ in range(16):
        cur=T[cur][x]
        if cur not in mp:mp[cur]=len(mp)
        pat.append(mp[cur])
    return tuple(pat)
def mul_inv(T,x,zero,one,idems):
    row=T[x];c=collections.Counter(row)
    return (len(c),c[zero],c[x],sum(v==y for y,v in enumerate(row)),int(x in idems),sum(v in idems for v in row),tuple(sorted(c.values())),power_pattern(T,one,x))
def canonical_mul_words():
    words={}
    for k in range(6):
        for j in range(8):
            for e in range(2):
                s=(pow(2,k,N)*pow(3,j,N)*(31 if e else 1))%N
                words.setdefault(s,(k,j,e))
    words[0]=None;assert len(words)==N;return words
MWORDS=canonical_mul_words()
def ppow(T,a,k,one):
    r=one
    for _ in range(k):r=T[r][a]
    return r
def mul_candidates(T):
    zero=next(x for x in range(N) if all(T[x][y]==x and T[y][x]==x for y in range(N)))
    one=identity(T,True);idems={x for x in range(N) if T[x][x]==x}
    sig={x:mul_inv(T,x,zero,one,idems) for x in range(N)}
    Tc=[[a*b%N for b in range(N)] for a in range(N)];csig={x:mul_inv(Tc,x,0,1,{0,1}) for x in range(N)}
    c2=[x for x in range(N) if sig[x]==csig[2]];c3=[x for x in range(N) if sig[x]==csig[3]];cm=[x for x in range(N) if sig[x]==csig[31]]
    out=[]
    for l2,l3,lm in itertools.product(c2,c3,cm):
        can_to_priv={0:zero}
        for s,w in MWORDS.items():
            if s==0:continue
            k,j,e=w;r=one
            r=T[r][ppow(T,l2,k,one)];r=T[r][ppow(T,l3,j,one)]
            if e:r=T[r][lm]
            can_to_priv[s]=r
        if len(set(can_to_priv.values()))<N:continue
        ok=True
        for a in range(N):
            for b in range(N):
                if can_to_priv[(a*b)%N]!=T[can_to_priv[a]][can_to_priv[b]]:ok=False;break
            if not ok:break
        if ok:
            q=[None]*N
            for s,l in can_to_priv.items():q[l]=s
            tq=tuple(q)
            if tq not in out:out.append(tq)
    assert len(out)==64,(len(out),len(c2),len(c3),len(cm));return out

def active_filter(cand,bridge):
    qs=0
    while len(cand)>1:
        best=None
        for l in range(N):
            buckets=collections.Counter(c[l] for c in cand);score=(len(buckets),-max(buckets.values()))
            if best is None or score>best[0]:best=(score,l)
        l=best[1];obs=bridge(l);cand=[c for c in cand if c[l]==obs];qs+=1
    return list(cand[0]),qs

def trial(seed,nprog=80):
    rng=random.Random(seed);ps={};Ts={};invs={}
    for op in OPS:
        p=list(range(N));rng.shuffle(p);ps[op]=p;Ts[op],invs[op]=mk(op,p)
    mr=min_rank(Ts['min']);maps={'min':[mr[l] for l in range(N)],'max':[max_rank(Ts['max'])[l] for l in range(N)]};q={'min':0,'max':0}
    def bridge(op,l):return mr[ps['min'][invs[op][l]]]
    for op,fn in [('add',add_coordinates),('sub',sub_coordinates)]:
        g,coord=fn(Ts[op]);u=bridge(op,g);m=[None]*N
        for l,k in coord.items():m[l]=(k*u)%N
        maps[op]=m;q[op]=1
    basis,coord=xor_coordinates(Ts['xor']);cols=[bridge('xor',l) for l in basis];m=[None]*N
    for l,c in coord.items():
        y=0
        for i,v in enumerate(cols):
            if (c>>i)&1:y^=v
        m[l]=y
    maps['xor']=m;q['xor']=5
    maps['mul'],q['mul']=active_filter(mul_candidates(Ts['mul']),lambda l:bridge('mul',l))
    exact={op:sum(maps[op][ps[op][s]]==s for s in range(N))/N for op in OPS}
    invmap={}
    for op,m in maps.items():
        z=[None]*N
        for l,s in enumerate(m):z[s]=l
        invmap[op]=z
    e2e={}
    for d in range(1,11):
        ok=0
        for _ in range(nprog):
            v=truth=rng.randrange(N)
            for __ in range(d):
                op=rng.choice(OPS);b=rng.randrange(N);lo=Ts[op][invmap[op][v]][invmap[op][b]];v=maps[op][lo];truth=f(op,truth,b)
            ok+=v==truth
        e2e[str(d)]=ok/nprog
    return {'queries':q,'mapping_exact':exact,'e2e':e2e}

def main():
    st=time.time();ts=[trial(40000+i) for i in range(100)];q={op:statistics.mean(t['queries'][op] for t in ts) for op in OPS};total=sum(q.values())
    out={'experiment':'structured_cross_source_gauge_alignment_v2','N':N,'trials':100,
      'common_coordinate':'MIN-chain rank inferred only from private MIN transition structure',
      'methods':{'min/max':'0 bridge queries via chain structure','add/sub':'1 symmetry-breaking generator query each','xor':'5 basis queries','mul':'enumerate 64 multiplication-monoid gauge candidates from private table; greedy active bridge collapses them in 3 queries'},
      'mean_bridge_queries':q,'total_queries_per_pool':total,'naive_nonreference_per_state_queries':5*N,'query_reduction_vs_naive':1-total/(5*N),
      'all_mappings_exact':all(all(v==1 for v in t['mapping_exact'].values()) for t in ts),'all_e2e_exact':all(all(v==1 for v in t['e2e'].values()) for t in ts),
      'mean_e2e':{str(d):statistics.mean(t['e2e'][str(d)] for t in ts) for d in range(1,11)},
      'interpretation':'private transition algebra reduces cross-source alignment to symmetry breaking: bridge query complexity tracks the residual automorphism/gauge degrees of freedom, not state-vocabulary size',
      'claim_boundary':'requires exhaustive source-local transition tables and a same-latent-state transfer bridge into MIN; operation family remains synthetic and known enough to construct structural coordinate candidates',
      'elapsed_s':time.time()-st}
    p=Path('/mnt/data/opfusion-local/evaluations/local_causal_saliency/structured_cross_source_gauge_alignment_v2.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));print('WROTE',p)
if __name__=='__main__':main()
