import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,json,time; sys.path.insert(0,ROOT+"/floret"); sys.path.insert(1,ROOT+"/fam")
from bridge import *
nm=sys.argv[1]; W=int(sys.argv[2]); M=int(sys.argv[3])
g_=build_bridged(W,M*2*PER+0.01,RULES[nm]); B=End(g_,40*W//2,2*PER)
fe,fm,bm,g=inv(B); qc=bm/fe; s=(g-fm)/fe
print(f"{nm} W={W} M={M}: f_exp={fe*10:.3f}/10 f_mid={fm*10:.3f}/10 b_min={bm*10:.3f}/10 g={g*10:.3f}/10  c_c={qc:.3f} s={s:+.4f}")
def e3(p,q):
    c=sorted([0,p,q]); return c[0]*(1-fe-fm)+c[1]*fm+c[2]*fe
def rpred(ang):
    th=math.radians(ang); lo,hi=0,20
    for _ in range(60):
        m_=(lo+hi)/2; p,q=m_*math.cos(th),m_*math.sin(th); c=sorted([0,p,q]); pp,qq=c[2]-c[0],c[1]-c[0]
        if pp<=qc+min(0,s)*qq+1e-12: lo=m_
        else: hi=m_
    return lo
pts=[]; t0=time.time()
for ang in [0,10,20,30,40,45,50,60,70,80,90,135,180,225]:
    th=math.radians(ang); c,s_=math.cos(th),math.sin(th)
    free=lambda r: B.e(r*c,r*s_)>=e3(r*c,r*s_)-1e-9
    lo,hi=0.0,0.5
    while free(hi) and hi<12: lo,hi=hi,hi*2
    for _ in range(26):
        m_=(lo+hi)/2
        if free(m_): lo=m_
        else: hi=m_
    pts.append((ang,lo*c,lo*s_)); 
    print(f"  {ang:3d}  exact ({lo*c:7.4f},{lo*s_:7.4f})  r={lo:.4f}   predicted r={rpred(ang):.4f}",flush=True)
print(f"[{time.time()-t0:.0f}s] non-stationary reads {B.bad}")
json.dump({"fe":fe,"fm":fm,"bm":bm,"g":g,"s":s,"qc":qc,"pts":pts},open(fROOT+"/floret/concave_{nm}_W{W}_M{M}.json","w"))
