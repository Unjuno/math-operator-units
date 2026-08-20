from __future__ import annotations
import argparse, importlib.util, json, random, sys, time
from pathlib import Path
import numpy as np
import torch

BASE=Path(__file__).with_name('experiment_joint_opaque_positionfree_heldout_transformer.py')
spec=importlib.util.spec_from_file_location('jointbase',BASE)
j=importlib.util.module_from_spec(spec); sys.modules['jointbase']=j; spec.loader.exec_module(j); h=j.h


def spectral_code(dim):
    adjacency=np.zeros((h.N,h.N),dtype=float)
    for op in h.SEEN_OPS:
        for a in range(h.N):
            for b in range(h.N):
                y=h.op_apply(op,a,b)
                for x in (a,b):
                    if x!=y: adjacency[x,y]+=1; adjacency[y,x]+=1
    inv=np.diag(1/np.sqrt(np.maximum(adjacency.sum(1),1e-9))); lap=np.eye(h.N)-inv@adjacency@inv; _,vectors=np.linalg.eigh(lap); e=vectors[:,1:dim+1]; e/=np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-9); return torch.tensor(e,dtype=torch.float32)


def decode(p1,p2,code):
    z=.6*(p1@code)+.4*(p2@code); z=z/z.norm().clamp_min(1e-9); return int((code@z).argmax())


def run(seed,device,nprog=60):
    rng=random.Random(seed+5555); codes=j.make_codes(rng); models,_,ads,_,_=j.train_pool(seed,device,220); _,threshold,routes=j.behavioral_code_grounding(codes,models,ads,device); code={d:spectral_code(d) for d in [8,12,16,24]}; cache={}
    def prob(si,a,b):
        key=(si,a,b)
        if key not in cache: cache[key]=h.action_logprobs(models[si],ads[si],a,b,device).exp()
        return cache[key]
    out={}; rrng=random.Random(seed+77000)
    for depth in range(1,11):
        correct={k:0 for k in ['explicit','sp8','sp12','sp16','sp24']}
        for _ in range(nprog):
            init=rrng.randrange(h.N); values={k:init for k in correct}; truth=init; program=[(rrng.choice(h.OPS),rrng.randrange(h.N)) for __ in range(depth)]
            for op,b in program:
                route=routes[codes[op]]; pri,sec=route['top'],route['second']; new={}
                for name,value in values.items():
                    p1,p2=prob(pri,value,b),prob(sec,value,b); y=int((.6*p1+.4*p2).argmax()) if name=='explicit' else decode(p1,p2,code[int(name[2:])]); new[name]=y
                values=new; truth=h.op_apply(op,truth,b)
            for name,value in values.items(): correct[name]+=value==truth
        out[str(depth)]={name:correct[name]/nprog for name in correct}
    return {'seed':seed,'threshold':threshold,'result':out}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); a=ap.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); start=time.time(); pools=[run(seed,device) for seed in a.seeds]; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps({'experiment':'spectral_transition_latent_source_fusion','embedding_source':'normalized Laplacian of seen-operation transition graph; held-out MUL excluded','pools':pools,'elapsed_s':time.time()-start},indent=2))

if __name__=='__main__': main()
