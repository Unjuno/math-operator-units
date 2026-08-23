from __future__ import annotations
import argparse, importlib.util, json, random, sys, time
from pathlib import Path
import torch

SEG=Path(__file__).with_name('experiment_learned_stream_stage_segmentation.py')
spec=importlib.util.spec_from_file_location('segbase',SEG)
g=importlib.util.module_from_spec(spec); sys.modules['segbase']=g; spec.loader.exec_module(g); j=g.j; h=g.h


def canonical_codes(): return {op:(f'A{i//3}',f'B{i%3}') for i,op in enumerate(h.OPS)}


def make_paraphrase(canon,seed):
    rng=random.Random(seed); first=['X0','X1']; second=['Y0','Y1','Y2']; rng.shuffle(first); rng.shuffle(second); amap={'A0':first[0],'A1':first[1]}; bmap={'B0':second[0],'B1':second[1],'B2':second[2]}; return {op:(amap[c[0]],bmap[c[1]]) for op,c in canon.items()}


def learn_component_alignment(canon,para):
    counts={}
    for op in h.SEEN_OPS:
        for qc,pc in zip(canon[op],para[op]): counts[(qc,pc)]=counts.get((qc,pc),0)+1
    inv={}
    for qc in sorted(set(q for q,_ in counts)):
        inv[max((n,pc) for (q,pc),n in counts.items() if q==qc)[1]]=qc
    return inv


def translate(tokens,inv): return [inv.get(t,t) for t in tokens]


def run(seed,device,steps=220,n=100):
    canon=canonical_codes(); para=make_paraphrase(canon,seed+4000); inv=learn_component_alignment(canon,para); held=para[h.HELDOUT_OP]; inferred=(inv.get(held[0]),inv.get(held[1])); models,_,ads,_,_=j.train_pool(seed,device,steps); _,threshold,routes=j.behavioral_code_grounding(canon,models,ads,device); segmenter=g.train_segmenter(); grounded=set(canon.values()); rng=random.Random(seed+6000); out={}
    for depth in range(1,11):
        seg=e2e=route=stages=0
        for _ in range(n):
            value=truth=rng.randrange(h.N); raw=[]; gold=[]; plan=[]
            for __ in range(depth):
                op=rng.choice(h.OPS); b=rng.randrange(h.N); v=rng.randrange(g.STAGE_VARIANTS); raw.extend(g.render_stage(para[op],b,v)); gold.append(len(raw)-1); plan.append((op,b))
            translated=translate(raw,inv); pred,_=g.predict_ends(segmenter,g.normalize_stream(translated,grounded)); seg+=pred==gold; chunks=g.decode_segments(translated,pred,grounded,routes)
            if chunks is None: continue
            for (code,b),(op,_) in zip(chunks,plan):
                r=routes[code]; route+=r['top']==h.OPS.index(op); stages+=1; value=h.run_source(models[r['top']],ads[r['top']],value,b,device); truth=h.op_apply(op,truth,b)
            e2e+=len(chunks)==depth and value==truth
        out[str(depth)]={'segmentation_exact':seg/n,'route_acc':route/max(stages,1),'e2e':e2e/n}
    return {'seed':seed,'component_inverse_map':inv,'heldout_paraphrase':held,'heldout_canonical_inferred':inferred,'heldout_mapping_ok':inferred==canon[h.HELDOUT_OP],'heldout_pair_never_seen_as_pair':held not in [para[o] for o in h.SEEN_OPS],'threshold':threshold,'result':out}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--seeds',type=int,nargs='+',default=[310,311,312]); a=ap.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1); device=torch.device('cpu'); start=time.time(); pools=[run(seed,device) for seed in a.seeds]; Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps({'experiment':'compositional_command_paraphrase_alignment','protocol':'canonical command is behaviorally grounded; paraphrase receives no behavior query; component substitution learned from paired unlabeled seen-command views; heldout paraphrase is an unseen component combination','pools':pools,'elapsed_s':time.time()-start},indent=2))

if __name__=='__main__': main()
