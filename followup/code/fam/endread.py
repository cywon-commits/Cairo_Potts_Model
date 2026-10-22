import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,json,time; sys.path.insert(0,ROOT+"/fam"); sys.path.insert(1,ROOT+"/tm")
from tm_eig import *
from analyze import invariants, ray_boundary
from hexcheck import pred_poly, ray_hit
from lattices import build_hub
class End:
    def __init__(s,graph,spc,key,K=4): s.n,s.E,s.xs=graph; s.spc=spc; s.key=key; s.K=K; s.bad=0
    def e(s,p,q):
        m=marks(s.n,s.E,s.xs,[0,p,q],s.key); K=s.K
        a=(m[-3]-m[-3-K])/K; b=(m[-3-K]-m[-3-2*K])/K
        if abs(a-b)>1e-9: s.bad+=1
        return a/s.spc
def g_of(B): h=1e-3; return (B.e(60.0,0.1+h)-B.e(60.0,0.1-h))/(2*h)
nm=sys.argv[1]; W=2; M=int(sys.argv[2]) if len(sys.argv)>2 else 160
ck=lambda x: round(x-0.2)
G={"hub m=4":(build_hub([30,90,210,270],W)(M*1.5+.01),7*W,lambda x: math.floor((x+1e-6)/1.5)),
   "Cairo":(cairo_cyl(W,M)[:3],6*W,ck),
   "Cairo + tail on Vp":(build_tail(W,M,'Vp'),7*W,ck),
   "Cairo + A-B bridge":(build_bridge(W,M),7*W,ck)}[nm]
B=End(*G); t=time.time()
fe,fm,bm=invariants(B); g=g_of(B); qc=bm/fe; s=(g-fm)/fe
angs=list(range(0,360,15)); pts=ray_boundary(B,fe,fm,angs,rmax=8)
poly=pred_poly(qc,s)
devs=[abs(math.hypot(p,q)-ray_hit(poly,a)) for a,p,q in pts]
print(f"{nm:20s} M={M} f_exp={fe:.5f} f_mid={fm:.5f} b_min={bm:.5f} g={g:.5f} q_c={qc:.3f} s={s:+.3f} | max dev={max(devs):.1e} (non-stationary reads: {B.bad}) [{time.time()-t:.0f}s]")
json.dump({"fe":fe,"fm":fm,"bm":bm,"g":g,"qc":qc,"s":s,"pts":pts,"poly":poly,"devs":devs},open(fROOT+"/fam/end_{nm.replace(' ','_')}_{M}.json","w"))
