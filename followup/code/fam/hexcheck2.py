import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,json; sys.path.insert(0,ROOT+"/fam"); sys.path.insert(1,ROOT+"/tm")
from tm_eig import *
from analyze import invariants, ray_boundary
from lattices import build_hub
from hexcheck import pred_poly, ray_hit

class EigHub(Eig):
    def e(s,p,q,check=False):
        m=marks(s.n,s.E,s.xs,[0,p,q],lambda x: math.floor((x+1e-6)/1.5))
        M=len(m); t0=M//2-s.K; return (m[t0+s.K]-m[t0])/s.K/s.spc
def g_of(B):
    h=1e-3; return (B.e(60.0,0.1+h)-B.e(60.0,0.1-h))/(2*h)
angs=list(range(0,360,15))
res={}
for W in [int(sys.argv[1])]:
    cases=[("hub m=4",EigHub(build_hub([30,90,210,270],W)(30*1.5+.01),7*W)),
           ("Cairo",Eig(cairo_cyl(W,40)[:3],6*W)),
           ("Cairo + tail on Vp",Eig(build_tail(W,40,'Vp'),7*W)),
           ("Cairo + A-B bridge",Eig(build_bridge(W,40),7*W))]
    print(f"--- W={W}")
    for nm,B in cases:
        fe,fm,bm=invariants(B); g=g_of(B); qc=bm/fe; s=(g-fm)/fe
        poly=pred_poly(qc,s); pts=ray_boundary(B,fe,fm,angs,rmax=8)
        dev=max(abs(math.hypot(p,q)-ray_hit(poly,a)) for a,p,q in pts)
        print(f"{nm:20s} f_exp={fe:.5f} f_mid={fm:.5f} b_min={bm:.5f} g={g:.5f} q_c={qc:.3f} s={s:+.3f} | max dev from predicted hexagon = {dev:.1e}",flush=True)
        res[f"{nm}|{W}"]={"fe":fe,"fm":fm,"bm":bm,"g":g,"qc":qc,"s":s,"pts":pts,"poly":poly}
json.dump(res,open(ROOT+"/fam/hexcheck2_W"+sys.argv[1]+".json","w"))
