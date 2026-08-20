from __future__ import annotations
import json,random,statistics,time
from pathlib import Path
N=32;M=N*N
BUDGETS=[4,8,16,24,32,64,128,256,512,1024]
def active_pairs(k):
    xs=[(i,i) for i in range(min(k,N))]
    if k>N:
        seen=set(xs)
        for a in range(N):
            for b in range(N):
                if (a,b) not in seen:xs.append((a,b));seen.add((a,b))
                if len(xs)>=k:return xs
    return xs

def factor_codec(rng):
    pa=list(range(N));pb=list(range(N));rng.shuffle(pa);rng.shuffle(pb)
    return pa,pb
def factor_token(codec,a,b):
    pa,pb=codec;return pa[a]*N+pb[b]
def arbitrary_codec(rng):
    p=list(range(M));rng.shuffle(p);return p
def arbitrary_token(p,a,b):return p[a*N+b]

def infer_factor(anchors):
    A={};B={}
    for a,b,t in anchors:A[a]=t//N;B[b]=t%N
    return A,B
def factor_coverage(A,B):return len(A)*len(B)/M

def program_resolve(mapped_pair,seed,nprog=200,depth=5):
    rng=random.Random(seed);ok=0;stages=0;known=0
    for _ in range(nprog):
        v=rng.randrange(N);good=True
        for __ in range(depth):
            b=rng.randrange(N);stages+=1
            if mapped_pair(v,b):known+=1
            else:good=False
            v=(v+b)%N
        ok+=good
    return ok/nprog,known/stages

def trial(seed):
    rng=random.Random(seed);fac=factor_codec(rng);arb=arbitrary_codec(rng);rows=[]
    allpairs=[(a,b) for a in range(N) for b in range(N)]
    for k in BUDGETS:
        ap=active_pairs(k)
        fa=[(a,b,factor_token(fac,a,b)) for a,b in ap];A,B=infer_factor(fa)
        fac_exact=sum((a in A and b in B and A[a]*N+B[b]==factor_token(fac,a,b)) for a,b in allpairs)/M
        amap={(a,b):arbitrary_token(arb,a,b) for a,b in ap};arb_cov=len(amap)/M
        fprog,fstage=program_resolve(lambda a,b:a in A and b in B,seed+k,100,5)
        aprog,astage=program_resolve(lambda a,b:(a,b) in amap,seed+k,100,5)
        rows.append({'budget':k,'factorized_pair_exact':fac_exact,'arbitrary_pair_known':arb_cov,'factorized_depth5_resolved':fprog,'arbitrary_depth5_resolved':aprog,'factorized_stage_coverage':fstage,'arbitrary_stage_coverage':astage})
    randomrows=[]
    for k in [8,16,32,64,128]:
        sample=rng.sample(allpairs,k);A,B=infer_factor([(a,b,factor_token(fac,a,b)) for a,b in sample]);randomrows.append({'budget':k,'a_states_seen':len(A),'b_states_seen':len(B),'pair_coverage':factor_coverage(A,B)})
    return rows,randomrows

def main():
    st=time.time();tr=[trial(60000+i) for i in range(500)]
    outrows=[]
    for bi,k in enumerate(BUDGETS):
        keys=tr[0][0][bi].keys();r={'budget':k}
        for key in keys:
            if key=='budget':continue
            r[key]=statistics.mean(x[0][bi][key] for x in tr)
        outrows.append(r)
    random=[]
    for i,k in enumerate([8,16,32,64,128]):
        random.append({'budget':k,'mean_a_states_seen':statistics.mean(x[1][i]['a_states_seen'] for x in tr),'mean_b_states_seen':statistics.mean(x[1][i]['b_states_seen'] for x in tr),'mean_pair_coverage':statistics.mean(x[1][i]['pair_coverage'] for x in tr)})
    out={'experiment':'input_abi_factorization_sample_complexity_v1','N':N,'trials':500,
      'factorized_codec':'token = privateA[a]*N + privateB[b], with independent random operand-state permutations; token quotient/remainder expose compositional slots but no semantic state meaning',
      'arbitrary_codec':'one random permutation over all N^2 semantic pairs',
      'active_anchor_results':outrows,'random_factorized_anchor_results':random,
      'key_result':'32 diagonal behavior anchors recover both 32-state component permutations and therefore all 1024 pair tokens exactly; an arbitrary pair codebook with the same 32 anchors exposes only 32/1024 pairs and has essentially zero safe depth-5 coverage',
      'identifiability_boundary':'without input factorization or a source-native encoder/query that maps semantic components to private input form, arbitrary N^2 pair codebooks require O(N^2) direct alignment evidence; compositional input ABI reduces this to O(N)',
      'elapsed_s':time.time()-st}
    p=Path('/mnt/data/opfusion-local/evaluations/local_causal_saliency/input_abi_factorization_v1.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));print('WROTE',p)
if __name__=='__main__':main()
