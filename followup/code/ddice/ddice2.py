import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,json; sys.path.insert(0,ROOT+"/fam"); sys.path.insert(1,ROOT+"/ddice")
import numpy as np
from lib import cylinder, frontier, a1, a2, A
from lattices import honey_points, dice_rule
from tm_eig import marks
PER=math.sqrt(3)/2
def build2(W,L,with_t=True):
    n,E,xs,typ,pos,disp=cylinder(honey_points(R=int(L)+14,hubs=lambda i,j:True),W*a2,L,dice_rule)
    kind=['K' if t=='H' else 'H' for t in typ]
    E2=set(); X=list(xs); K2=list(kind); m=n
    for a,b in E:
        h,k=(a,b) if kind[a]=='H' else (b,a)
        e=m; m+=1; X.append(float(xs[h]+(xs[k]-xs[h])/2)); K2.append('e')
        p=m; m+=1; X.append(float(xs[h]+(xs[k]-xs[h])/4)); K2.append('P')
        E2|={(min(h,e),max(h,e)),(min(k,e),max(k,e)),(min(e,p),max(e,p)),(min(h,p),max(h,p))}
    if with_t:
        Ks=[v for v in range(n) if kind[v]=='K']
        for u in Ks:
            for v in Ks:
                d=disp(u,v)
                if np.allclose(d,a1,atol=1e-6):
                    t=m; m+=1; X.append(float((xs[u]+xs[v])/2)); K2.append('t')
                    E2|={(min(u,t),max(u,t)),(min(v,t),max(v,t))}
    return m,sorted(E2),np.array(X),K2
class End:
    def __init__(s,W,M,spc,K=4):
        s.n,s.E,s.xs,s.kind=build2(W,M*PER+0.01); s.K=K; s.bad=0; s.spc=spc
    def e(s,p,q):
        m=marks(s.n,s.E,s.xs,[0,p,q],lambda x: math.floor((x+1e-6)/PER)); K=s.K
        a=(m[-3]-m[-3-K])/K; b=(m[-3-K]-m[-3-2*K])/K
        if abs(a-b)>1e-9: s.bad+=1
        return a/s.spc
if __name__=="__main__":
    from collections import Counter
    W=int(sys.argv[1]); M=int(sys.argv[2])
    B=End(W,M,16*W)
    deg=Counter()
    for a,b in B.E: deg[a]+=1; deg[b]+=1
    mid=[v for v in range(B.n) if M*PER*0.4<B.xs[v]<M*PER*0.6]
    print("bulk (kind,deg):",sorted(Counter((B.kind[v],deg[v]) for v in mid).items()),"frontier",frontier(B.n,B.E,B.xs))
    h=1e-3
    fe=(B.e(0,0.1+h)-B.e(0,0.1-h))/(2*h); fm=(B.e(0.1+h,0.3)-B.e(0.1-h,0.3))/(2*h)
    bm=B.e(0,60.0); g=(B.e(60,0.1+h)-B.e(60,0.1-h))/(2*h); qc=bm/fe; s=(g-fm)/fe
    print(f"f_exp={fe*16:.3f}/16 f_mid={fm*16:.3f}/16 b_min={bm*16:.3f}/16 g={g*16:.3f}/16 | b/f={qc:.3f} s={s:+.3f}  bound 1-b/f={1-qc:+.3f}  d={qc/(1-min(0,s)):.3f}")
    def e3(p,q):
        c=sorted([0,p,q]); return c[0]*(1-fe-fm)+c[1]*fm+c[2]*fe
    pts=[]
    for ang in range(0,360,15):
        th=math.radians(ang); c,s_=math.cos(th),math.sin(th)
        free=lambda r: B.e(r*c,r*s_)>=e3(r*c,r*s_)-1e-9
        lo,hi=0.0,0.5
        while free(hi) and hi<12: lo,hi=hi,hi*2
        for _ in range(26):
            mm=(lo+hi)/2
            if free(mm): lo=mm
            else: hi=mm
        pts.append((ang,lo*c,lo*s_))
        cc=sorted([0,lo*c,lo*s_])
        if ang<=90 or ang in (135,180): print(f"  {ang:3d} ({lo*c:7.4f},{lo*s_:7.4f})  (p',q')=({cc[2]-cc[0]:.4f},{cc[1]-cc[0]:.4f})  C2: 3-2q'={3-2*(cc[1]-cc[0]):.4f}",flush=True)
    print("non-stationary reads:",B.bad)
    json.dump({"fe":fe,"fm":fm,"bm":bm,"g":g,"pts":pts},open(fROOT+"/ddice/ddice2_W{W}_M{M}.json","w"))
