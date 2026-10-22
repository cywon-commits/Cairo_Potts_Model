import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,time; sys.path.insert(0,ROOT+"/fam"); sys.path.insert(1,ROOT+"/ddice")
import numpy as np
from scipy.special import logsumexp
from lib import frontier
from ddice2 import build2, PER
def logZ_marks(n,E,xs,D,T,cellkey):
    order=sorted(range(n),key=lambda u:(round(xs[u],6),u)); rank={u:k for k,u in enumerate(order)}
    adj=[[] for _ in range(n)]
    for a,b in E: adj[a].append(b); adj[b].append(a)
    last=[max([rank[w] for w in adj[v]]+[rank[v]]) for v in range(n)]
    D=np.asarray(D,float); EQ=np.eye(3); fr=[]; vals=np.zeros(()); out=[]; shift=0.0
    for idx,u in enumerate(order):
        vals=vals[...,None]-D.reshape([1]*vals.ndim+[3])/T; k=vals.ndim
        for v in adj[u]:
            if v in fr:
                i=fr.index(v); sh=[1]*k; sh[i]=3; sh[k-1]=3; vals=vals-EQ.reshape(sh)/T
        fr.append(u)
        for i in reversed(range(len(fr))):
            if last[fr[i]]<=idx: vals=logsumexp(vals,axis=i); fr.pop(i)
        mx=vals.max(); vals=vals-mx; shift+=mx
        if idx+1==len(order) or cellkey(xs[order[idx+1]])!=cellkey(xs[u]):
            out.append(shift+float(logsumexp(vals)))
    return out
class FT:
    def __init__(s,W,M,with_t,spc):
        if with_t: s.g=build2(W,M*PER+0.01,True)
        else: s.g=build2(W,M*PER+0.01,False)
        s.spc=spc
    def f(s,p,q,T,K=4):
        n,E,xs,_=s.g
        m=logZ_marks(n,E,xs,[0,p,q],T,lambda x: math.floor((x+1e-6)/PER))
        return -T*(m[-3]-m[-3-K])/K/s.spc
if __name__=="__main__":
    which=sys.argv[1]; W=int(sys.argv[2]); M=int(sys.argv[3])
    with_t=(which=="s2"); spc=(16 if with_t else 15)*W
    B=FT(W,M,with_t,spc)
    n,E,xs,_=B.g; print("N",n,"frontier",frontier(n,E,xs),flush=True)
    if with_t: nP,nM,d=9/16,3/16,1.0; slope=0.0
    else: nP,nM,d=8/15,3/15,1.2; slope=-math.log(2)/5
    target=(nP+nM)/2
    for T in [float(x) for x in sys.argv[4].split(",")]:
        t0=time.time()
        def ncost(t,h=1e-3): return (B.f(t+h,t+h,T)-B.f(t-h,t-h,T))/(2*h)/2*2   # d f/dt along p=q=t gives n2+n3
        lo,hi=d-0.4,d+0.3
        for _ in range(14):
            mid=(lo+hi)/2
            if ncost(mid)>target: lo=mid
            else: hi=mid
        tc=(lo+hi)/2
        print(f"T={T:.2f}: mirror crossing t*={tc:.4f}   prediction d+slope*T={d+slope*T:.4f}   [{time.time()-t0:.0f}s]",flush=True)
