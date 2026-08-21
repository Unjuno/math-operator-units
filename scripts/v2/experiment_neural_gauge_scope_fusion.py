from __future__ import annotations
import json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
SEED=20260821
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def f0(x): return 0.55*x[:,0] + 0.20*x[:,1] + 0.10*np.sin(2*x[:,0])
def f1(x): return -0.25*x[:,0] + 0.65*x[:,1] + 0.12*np.sin(1.5*x[:,1])
def f2(x): return 0.30*x[:,0]*x[:,1] + 0.25*x[:,0] - 0.15*x[:,1]
def f3(x): return 0.22*(x[:,0]**2-x[:,1]**2) + 0.40*x[:,0] + 0.10*x[:,1]
FUNS=[f0,f1,f2,f3]

class Net(nn.Module):
    def __init__(self, seed):
        super().__init__(); torch.manual_seed(seed)
        self.net=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(-1)

def train_net(fn,seed):
    m=Net(seed); opt=torch.optim.Adam(m.parameters(),lr=1e-2)
    x=torch.rand(1600,2)*2-1; y=torch.tensor(fn(x.numpy()),dtype=torch.float32)
    for _ in range(260):
        opt.zero_grad(); loss=((m(x)-y)**2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        xt=torch.rand(1200,2)*2-1; yt=torch.tensor(fn(xt.numpy()),dtype=torch.float32)
        rmse=float(torch.sqrt(((m(xt)-yt)**2).mean()))
    return m,rmse

def h(y,k): return np.sinh(k*y)/np.sinh(k)
def hinv(z,k): return np.arcsinh(z*np.sinh(k))/k

def pct(a,p): return float(np.percentile(np.asarray(a),p))

def main():
    t0=time.time(); models=[]; rms=[]
    for i,fn in enumerate(FUNS):
        m,r=train_net(fn,1000+i); models.append(m); rms.append(r)
    rng=np.random.default_rng(SEED+7); N=5000; w=.6
    x=torch.tensor(rng.uniform(-1,1,(N,2)),dtype=torch.float32)
    ia=rng.integers(0,len(models),N); ib=(ia+1+rng.integers(0,len(models)-1,N))%len(models)
    ys=np.empty((len(models),N),dtype=np.float64)
    with torch.no_grad():
        for j,m in enumerate(models): ys[j]=m(x).numpy().astype(np.float64)
    ya=ys[ia,np.arange(N)]; yb=ys[ib,np.arange(N)]; truth=w*ya+(1-w)*yb
    common_err=[]; independent_err=[]; aligned_err=[]; mono_err=[]; grounded_err=[]
    for i in range(N):
        s=float(rng.uniform(.2,3)); o=float(rng.uniform(-2,2)); za=s*ya[i]+o; zb=s*yb[i]+o
        common_err.append(abs((w*za+(1-w)*zb-o)/s-truth[i]))
    for i in range(N):
        sa,sb=float(rng.uniform(.2,3)),float(rng.uniform(.2,3)); oa,ob=float(rng.uniform(-2,2)),float(rng.uniform(-2,2))
        za=sa*ya[i]+oa; zb=sb*yb[i]+ob; independent_err.append(abs(w*za+(1-w)*zb-truth[i]))
    for i in range(N):
        sa,sb=float(rng.uniform(.2,3)),float(rng.uniform(.2,3)); oa,ob=float(rng.uniform(-2,2)),float(rng.uniform(-2,2)); a1,a2=-.6,.7
        za1,za2=sa*a1+oa,sa*a2+oa; zb1,zb2=sb*a1+ob,sb*a2+ob
        sha=(za2-za1)/(a2-a1); oha=za1-sha*a1; shb=(zb2-zb1)/(a2-a1); ohb=zb1-shb*a1
        za=sa*ya[i]+oa; zb=sb*yb[i]+ob; aya=(za-oha)/sha; ayb=(zb-ohb)/shb
        aligned_err.append(abs(w*aya+(1-w)*ayb-truth[i]))
    for i in range(N):
        k=float(rng.uniform(.35,3.2)); za=h(ya[i],k); zb=h(yb[i],k); z=w*za+(1-w)*zb
        mono_err.append(abs(hinv(z,k)-truth[i]))
    kgrid=np.linspace(.35,3.2,1801); anchor=.37; pred=np.sinh(kgrid*anchor)/np.sinh(kgrid)
    for i in range(N):
        k=float(rng.uniform(.35,3.2)); obs=h(anchor,k); kh=float(kgrid[int(np.argmin(np.abs(pred-obs)))])
        za=h(ya[i],k); zb=h(yb[i],k); aya=hinv(za,kh); ayb=hinv(zb,kh)
        grounded_err.append(abs(w*aya+(1-w)*ayb-truth[i]))
    out={
      'experiment':'v2_phase5_actual_neural_gauge_scope_fusion','neural_models':{'count':len(models),'test_rmse':rms},'trials':N,'weight':w,
      'common_positive_affine':{'composer_decision':'resolved','mean_semantic_error':float(np.mean(common_err)),'p95':pct(common_err,95),'max':float(np.max(common_err))},
      'independent_positive_affine_without_alignment':{'correct_composer_decision':'unresolved','mean_native_mean_error_vs_semantic':float(np.mean(independent_err)),'p95':pct(independent_err,95),'max':float(np.max(independent_err)),'finding':'gauge group alone is insufficient; coupling/scope across plugin outputs matters'},
      'independent_positive_affine_plus_two_anchors_per_plugin':{'composer_decision':'resolved_after_alignment','grounding_cost_per_plugin':2,'mean_semantic_error':float(np.mean(aligned_err)),'p95':pct(aligned_err,95),'max':float(np.max(aligned_err))},
      'common_strict_monotone_without_geometry':{'correct_composer_decision':'unresolved','mean_if_naively_fused':float(np.mean(mono_err)),'p95':pct(mono_err,95),'max':float(np.max(mono_err))},
      'common_one_parameter_monotone_plus_one_interior_anchor':{'composer_decision':'resolved_after_geometry_grounding','grounding_cost':1,'mean_semantic_error':float(np.mean(grounded_err)),'p95':pct(grounded_err,95),'max':float(np.max(grounded_err))},
      'design_update':{'old':'capability = f(gauge_group)','new':'capability = f(gauge_group, gauge_scope_or_coupling, available_chart_alignment)','reason':'weighted mean is equivariant to a shared positive-affine gauge, not to independent positive-affine gauges on each plugin output'},'elapsed_s':time.time()-t0}
    out['pass']=bool(max(common_err)<1e-10 and np.mean(independent_err)>0.1 and max(aligned_err)<1e-10 and np.mean(mono_err)>1e-4 and np.mean(grounded_err)<5e-4)
    Path('phase5_neural_gauge_scope.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
