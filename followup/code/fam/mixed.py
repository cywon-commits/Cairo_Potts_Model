import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,json,time; sys.path.insert(0,ROOT+"/fam"); sys.path.insert(1,ROOT+"/tm")
import numpy as np
from generic import cairo_cyl
from tm_eig import marks
from analyze import invariants, ray_boundary

PATTERNS={"x0":lambda i,j:False, "x1":lambda i,j:True,
          "x1/4":lambda i,j:i%4==0, "x1/2stripe":lambda i,j:i%2==0,
          "x1/2checker":lambda i,j:(i+j)%2==0, "x3/4":lambda i,j:i%4!=0}
def build_mixed(W,M,isbeta):
    n,E,xs,sites=cairo_cyl(W,M); E=set(E); xs=list(xs)
    for (t,i,j),idx in list(sites.items()):
        if t!='A' or ('B',i,j) not in sites: continue
        if isbeta(i,j):
            u,v=idx,sites[('B',i,j)]
        else:
            if ('Vp',i,j) not in sites or ('Hp',i,j) not in sites: continue
            u,v=sites[('Vp',i,j)],sites[('Hp',i,j)]
        xs.append(xs[idx]+0.25); E.add((u,n)); E.add((v,n)); n+=1
    return n,sorted(E),np.array(xs)
class End:
    def __init__(s,graph,spc,K=4): s.n,s.E,s.xs=graph; s.spc=spc; s.K=K; s.bad=0
    def e(s,p,q):
        m=marks(s.n,s.E,s.xs,[0,p,q],lambda x: round(x-0.2)); K=s.K
        a=(m[-3]-m[-3-K])/K; b=(m[-3-K]-m[-3-2*K])/K
        if abs(a-b)>1e-9: s.bad+=1
        return a/s.spc
def pred_poly(qc,s):
    d=qc/(1-min(0,s)); return [(qc,0),(d,d),(0,qc),(-d,0),(-qc,-qc),(0,-d)]
def ray_hit(poly,ang):
    th=math.radians(ang); u=(math.cos(th),math.sin(th)); best=None
    for k in range(6):
        (x1,y1),(x2,y2)=poly[k],poly[(k+1)%6]
        a,b=u[0],-(x2-x1); c,d=u[1],-(y2-y1); det=a*d-b*c
        if abs(det)<1e-12: continue
        r=(x1*d-b*y1)/det; t=(a*y1-c*x1)/det
        if r>0 and -1e-9<=t<=1+1e-9: best=r if best is None else min(best,r)
    return best
if __name__=="__main__":
    nm=sys.argv[1]; W=int(sys.argv[2]); M=int(sys.argv[3]) if len(sys.argv)>3 else 160
    B=End(build_mixed(W,M,PATTERNS[nm]),7*W); t=time.time()
    fe,fm,bm=invariants(B); h=1e-3; g=(B.e(60,0.1+h)-B.e(60,0.1-h))/(2*h)
    qc=bm/fe; s=(g-fm)/fe
    angs=list(range(0,360,15)); pts=ray_boundary(B,fe,fm,angs,rmax=8); poly=pred_poly(qc,s)
    devs=[abs(math.hypot(p,q)-ray_hit(poly,a)) for a,p,q in pts]
    print(f"{nm:12s} W={W} M={M}: f_exp={fe:.5f} f_mid={fm:.5f} b_min={bm:.5f} g={g:.5f} | q_c={qc:.4f} s={s:+.4f} | max dev={max(devs):.1e} [{time.time()-t:.0f}s]",flush=True)
    json.dump({"fe":fe,"fm":fm,"bm":bm,"g":g,"qc":qc,"s":s,"pts":pts,"devs":devs},open(fROOT+"/fam/mixed_{nm.replace('/','_')}_W{W}.json","w"))
