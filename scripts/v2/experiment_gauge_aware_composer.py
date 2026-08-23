from __future__ import annotations
from dataclasses import dataclass
import json,math,random
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class GaugeCertificate:
    group: str
    scope: str

@dataclass
class ComposeResult:
    status: str
    value: float|None
    reason: str
    grounding_used: int=0

class GaugeAwareComposer:
    CAPABILITIES={
      'full_permutation': {'select_a','select_b'},
      'strict_monotone': {'select_a','select_b','min','max'},
      'positive_affine': {'select_a','select_b','min','max','weighted_mean'},
      'identity': {'select_a','select_b','min','max','weighted_mean'},
    }
    def compose(self,a,b,op,cert:GaugeCertificate,w=.6,ground_geometry=None):
        if op in self.CAPABILITIES.get(cert.group,set()):
            return ComposeResult('resolved',self._apply(a,b,op,w),'gauge_equivariant_operation')
        if ground_geometry is not None:
            grounded=ground_geometry(cert)
            if grounded is not None:
                newcert,to_chart,from_chart,cost=grounded
                if op in self.CAPABILITIES.get(newcert.group,set()):
                    aa,bb=to_chart(a),to_chart(b);z=self._apply(aa,bb,op,w)
                    return ComposeResult('resolved',from_chart(z),'additional_geometry_grounded',cost)
        return ComposeResult('unresolved',None,'fusion_not_identifiable_under_residual_gauge')
    @staticmethod
    def _apply(a,b,op,w):
        if op=='select_a':return a
        if op=='select_b':return b
        if op=='min':return min(a,b)
        if op=='max':return max(a,b)
        if op=='weighted_mean':return w*a+(1-w)*b
        raise KeyError(op)

def h(y,k): return math.sinh(k*y)/math.sinh(k)
def hinv(z,k): return math.asinh(z*math.sinh(k))/k

def main():
    rng=random.Random(31415);C=GaugeAwareComposer();N=5000;w=.6
    perm_mean_unres=perm_proj_res=0
    for _ in range(N):
        a,b=rng.randrange(10),rng.randrange(10)
        perm_mean_unres += C.compose(a,b,'weighted_mean',GaugeCertificate('full_permutation','labels')).status=='unresolved'
        perm_proj_res += C.compose(a,b,'select_a',GaugeCertificate('full_permutation','labels')).status=='resolved'
    mono_mean_unres=mono_min_res=0
    for _ in range(N):
        a,b=rng.uniform(-1,1),rng.uniform(-1,1)
        mono_mean_unres += C.compose(a,b,'weighted_mean',GaugeCertificate('strict_monotone','R')).status=='unresolved'
        mono_min_res += C.compose(a,b,'min',GaugeCertificate('strict_monotone','R')).status=='resolved'
    aff_err=[]
    for _ in range(N):
        a,b=rng.uniform(-1,1),rng.uniform(-1,1);s=rng.uniform(.2,3);o=rng.uniform(-2,2)
        semantic=C.compose(a,b,'weighted_mean',GaugeCertificate('positive_affine','R'),w=w).value
        native=C.compose(s*a+o,s*b+o,'weighted_mean',GaugeCertificate('positive_affine','R'),w=w).value
        aff_err.append(abs((native-o)/s-semantic))
    ground_err=[];ground_cost=[];unres_without=0
    for _ in range(N):
        truek=rng.uniform(.2,3.5);a,b=rng.uniform(-1,1),rng.uniform(-1,1)
        na,nb=h(a,truek),h(b,truek)
        unres_without += C.compose(na,nb,'weighted_mean',GaugeCertificate('strict_monotone','normalized_sinh_family'),w=w).status=='unresolved'
        obs=h(.37,truek)
        def ground(cert):
            ks=np.linspace(.2,3.5,1001);pred=np.sinh(ks*.37)/np.sinh(ks);est=float(ks[int(np.argmin(abs(pred-obs)))])
            return GaugeCertificate('identity','recovered_semantic_chart'), lambda z: hinv(z,est), lambda z: h(z,est), 1
        r=C.compose(na,nb,'weighted_mean',GaugeCertificate('strict_monotone','normalized_sinh_family'),w=w,ground_geometry=ground)
        truth=w*a+(1-w)*b
        ground_err.append(abs(hinv(r.value,truek)-truth));ground_cost.append(r.grounding_used)
    out={'experiment':'v2_phase4_gauge_aware_composer',
      'full_permutation':{'weighted_mean_unresolved':int(perm_mean_unres),'trials':N,'projection_resolved':int(perm_proj_res)},
      'strict_monotone':{'weighted_mean_unresolved':int(mono_mean_unres),'min_resolved':int(mono_min_res),'trials':N},
      'positive_affine':{'weighted_mean_resolved':N,'max_equivariance_error':max(aff_err),'mean_error':sum(aff_err)/N},
      'monotone_plus_minimal_geometry_grounding':{'without_grounding_unresolved':int(unres_without),'with_grounding_resolved':N,
        'mean_grounding_cost':sum(ground_cost)/N,'mean_semantic_fusion_error':sum(ground_err)/N,'p95_error':float(np.percentile(ground_err,95))},
      'pass':perm_mean_unres==N and perm_proj_res==N and mono_mean_unres==N and mono_min_res==N and max(aff_err)<1e-10 and max(ground_cost)==1,
      'design':'composer consumes residual gauge certificate; unsupported fusion is UNRESOLVED unless a grounding provider returns a stronger chart'}
    p=Path('phase4_gauge_aware_composer.json');p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
