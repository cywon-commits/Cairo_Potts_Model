import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,time; sys.path.insert(0,ROOT+"/fam")
from lattices import *

def invariants(B):
    h=1e-3
    fe=(B.e(0,0.1+h)-B.e(0,0.1-h))/(2*h)               # cost only on colour 3
    fm=(B.e(0.1+h,0.3)-B.e(0.1-h,0.3))/(2*h)           # colour 2 = middle cost
    bmin=B.e(0,60.0)                                    # colour 3 priced out
    return fe,fm,bmin

def e3(p,q,fe,fm):
    c=sorted([0.0,p,q]); return c[0]*(1-fe-fm)+c[1]*fm+c[2]*fe

def ray_boundary(B,fe,fm,angles,rmax=8):
    out=[]
    for ang in angles:
        th=math.radians(ang); c,s=math.cos(th),math.sin(th)
        free=lambda r: B.e(r*c,r*s)>=e3(r*c,r*s,fe,fm)-1e-9
        lo,hi=0.0,0.25
        while free(hi) and hi<rmax: lo,hi=hi,hi*2
        for _ in range(24):
            mid=(lo+hi)/2
            if free(mid): lo=mid
            else: hi=mid
        out.append((ang,lo*c,lo*s))
    return out

def report(name,B,angles=(0,30,45,60,90,120,135,150,180,225,270,315)):
    t=time.time()
    fe,fm,bm=invariants(B)
    print(f"\n=== {name} ===  f_exp={fe:.6f}  f_mid={fm:.6f}  b_min={bm:.6f}  b_min/f_exp={bm/fe:.6f}")
    for ang,p,q in ray_boundary(B,fe,fm,angles):
        print(f"   {ang:3d}deg  boundary (p,q)=({p:8.4f},{q:8.4f})")
    print(f"   [{time.time()-t:.1f}s]")
    return fe,fm,bm

if __name__=="__main__":
    for W in [2,3]:
        report(f"truncated honeycomb W={W}", Bulk(build_trunc('honey',W),5.0,9.0))
    report("truncated dice W=2 (pipeline cross-check)", Bulk(build_trunc('dice',2),5.0,9.0))
