from __future__ import annotations
import argparse, importlib.util, json, random, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

BASE=Path(__file__).with_name('experiment_joint_opaque_positionfree_heldout_transformer.py')
spec=importlib.util.spec_from_file_location('jointbase',BASE)
j=importlib.util.module_from_spec(spec); sys.modules['jointbase']=j; spec.loader.exec_module(j); h=j.h


def learn_code(dim=6,seed=606,steps=450):
    torch.manual_seed(seed); zparam=torch.nn.Parameter(torch.randn(h.N,dim)); opt=torch.optim.Adam([zparam],lr=.05); ii=torch.arange(h.N).repeat_interleave(h.N); jj=torch.arange(h.N).repeat(h.N)
    for _ in range(steps):
        z=F.normalize(zparam,dim=1); mix=F.normalize(.6*z[ii]+.4*z[jj],dim=1); loss=F.cross_entropy(mix@z.T/.05,ii); opt.zero_grad(); loss.backward(); opt.step()
    z=F.normalize(zparam.detach(),dim=1); mix=F.normalize(.6*z[ii]+.4*z[jj],dim=1); acc=float(((mix@z.T).argmax(1)==ii).float().mean()); return z,acc


def latent_decode(p1,p2,code):
    z=.6*(p1@code)+.4*(p2@code); z=z/z.norm().clamp_min(1e-9); return int((code@z).argmax())


def run(seed,device,nprog=60):
    code,pairacc=learn_code(); rng=random.Random(seed+5555); codes=j.make_codes(rng); models,_,ads,_,_=j.train_pool(seed,device,220); _,threshold,routes=j.behavioral_code_grounding(codes,models,ads,device); cache={}
    def prob(si,a,b):
        key=(si,a,b)
        if key not in cache: cache[key]=h.action_logprobs(models[si],ads[si],a,b,device).exp()
        return cache[key]
    out={}; rrng=random.Random(seed+88000)
    for depth in range(1,11):
        correct={'explicit':0,'learned6':0}; local={'explicit':0,'learned6':0}; stages=0
        for _ in range(nprog):
            init=rrng.randrange(h.N); values={'explicit':init,'learned6':init}; truth=init; program=[(rrng.choice(h.OPS),rrng.randrange(h.N)) for __ in range(depth)]
            for op,b in program:
                route=routes[codes[op]]; pri,sec=route['top'],route['second']; new={}; stages+=1
                for name,value in values.items():
                    p1,p2=prob(pri,value,b),prob(sec,value,b); y=int((.6*p1+.4*p2).argmax()) if name=='explicit' else latent_decode(p1,p2,code); local[name]+=y==h.op_apply(op,value,b); new[name]=y
                values=new; truth=h.op_apply(op,truth,b)
            for name,value in values.items(): correct[name]+=value==truth
        out[str(depth)]={name:{'e2e':correct[name]/nprog,'stage_local':local[name]/stages} for name in correct}
    return {'seed':seed,'threshold':threshold,'code_pairwise_06_decode':pairacc,'result':out}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); a=ap.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); start=time.time(); pools=[run(seed,device) for seed in a.seeds]; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps({'experiment':'fusion_safe_learned6_latent_source_fusion','code_learning':'state identity + desired 0.6/0.4 dominant-mixture decoding only; no operator/transition semantics','pools':pools,'elapsed_s':time.time()-start},indent=2))

if __name__=='__main__': main()
