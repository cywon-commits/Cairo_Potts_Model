import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/fam"); sys.path.insert(1,ROOT+"/tm")
import numpy as np
from generic import cairo_cyl
from cairo_pend import build as build_tail
from cairo_bridge import build_bridge

def marks(n,E,xs,D,cellkey):
    order=sorted(range(n),key=lambda u:(round(xs[u],6),u)); rank={u:k for k,u in enumerate(order)}
    adj=[[] for _ in range(n)]
    for a,b in E: adj[a].append(b); adj[b].append(a)
    last=[max([rank[w] for w in adj[v]]+[rank[v]]) for v in range(n)]
    D=np.asarray(D,float); EQ=np.eye(3); fr=[]; vals=np.zeros(()); out=[]
    for idx,u in enumerate(order):
        vals=vals[...,None]+D.reshape([1]*vals.ndim+[3]); k=vals.ndim
        for v in adj[u]:
            if v in fr:
                i=fr.index(v); sh=[1]*k; sh[i]=3; sh[k-1]=3; vals=vals+EQ.reshape(sh)
        fr.append(u)
        for i in reversed(range(len(fr))):
            if last[fr[i]]<=idx: vals=vals.min(axis=i); fr.pop(i)
        if idx+1==len(order) or cellkey(xs[order[idx+1]])!=cellkey(xs[u]): out.append(float(vals.min()))
    return out

class Eig:
    """bulk energy per site = min-plus eigenvalue of the column transfer matrix (periodic regime)."""
    def __init__(s,graph,sites_per_cell,K=4):
        s.n,s.E,s.xs=graph; s.spc=sites_per_cell; s.K=K
    def e(s,p,q,check=False):
        m=marks(s.n,s.E,s.xs,[0,p,q],lambda x: round(x-0.2))
        M=len(m); t0=M//2-s.K; incs=[(m[t+s.K]-m[t])/s.K for t in range(t0-6,t0+1)]
        if check and max(incs)-min(incs)>1e-9: print("   not stationary:",incs)
        return incs[-1]/s.spc

if __name__=="__main__":
    W=2
    lat={"Cairo":(cairo_cyl(W,40)[:3],6*W),
         "Cairo + A-B bridge":(build_bridge(W,40),7*W),
         "Cairo + tail on Vp":(build_tail(W,40,'Vp'),7*W)}
    for nm,(g,spc) in lat.items():
        B=Eig(g,spc)
        print(nm)
        for p,q in [(1.623,0.435),(0.0,0.1),(60,0.1),(1.0,1.0)]:
            print(f"   e({p},{q}) = {B.e(p,q,check=True):.6f}")
