import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,itertools; sys.path.insert(0,ROOT+"/fam")
import numpy as np
from scipy.special import logsumexp
from lib import cylinder, frontier, a1, a2
from lattices import honey_points, dice_rule
PER=math.sqrt(3)/2
def unit_tables(D,T):
    """exact sum (T>0) / min (T=0) over (e,P) of one unit, given colours (h,k). returns 3x3 array of log-weights (T>0) or energies (T=0)."""
    out=np.zeros((3,3))
    for h in range(3):
        for k in range(3):
            vals=[]
            for e in range(3):
                for p in range(3):
                    E=D[e]+D[p]+(e==h)+(e==k)+(e==p)+(p==h)
                    vals.append(E)
            vals=np.array(vals,float)
            out[h,k]= vals.min() if T==0 else logsumexp(-vals/T)
    return out
def tsite_table(D,T):
    out=np.zeros((3,3))
    for k1 in range(3):
        for k2 in range(3):
            vals=np.array([D[t]+(t==k1)+(t==k2) for t in range(3)],float)
            out[k1,k2]= vals.min() if T==0 else logsumexp(-vals/T)
    return out
class Dec:
    def __init__(s,W,M,m,with_t=True):
        n,E,xs,typ,pos,disp=cylinder(honey_points(R=M+14,hubs=lambda i,j:True),W*a2,M*PER+0.01,dice_rule)
        s.kind=['K' if t=='H' else 'H' for t in typ]
        s.n=n; s.xs=xs; s.m=m; s.W=W
        s.bonds=[]
        for a,b in E:
            h,k=(a,b) if s.kind[a]=='H' else (b,a); s.bonds.append((h,k,'u'))
        if with_t:
            Ks=[v for v in range(n) if s.kind[v]=='K']
            for u in Ks:
                for v in Ks:
                    if np.allclose(disp(u,v),a1,atol=1e-6): s.bonds.append((u,v,'t'))
        s.N=(4+12*m)*W
    def run(s,D,T):
        D=np.asarray(D,float)
        U=unit_tables(D,T); Tt=tsite_table(D,T)
        n=s.n; xs=s.xs
        order=sorted(range(n),key=lambda u:(round(xs[u],6),u)); rank={u:k for k,u in enumerate(order)}
        adj=[[] for _ in range(n)]
        for a,b,kind in s.bonds:
            M_=s.m*U if kind=='u' else Tt
            adj[a].append((b,M_)); adj[b].append((a,M_.T))
        last=[max([rank[w] for w,_ in adj[v]]+[rank[v]]) for v in range(n)]
        fr=[]; vals=np.zeros(()); marks=[]; shift=0.0
        key=lambda x: math.floor((x+1e-6)/PER)
        site=-D if T>0 else D
        if T>0: site=-D/T
        for idx,u in enumerate(order):
            vals=vals[...,None]+site.reshape([1]*vals.ndim+[3]); k=vals.ndim
            for v,M_ in adj[u]:
                if v in fr:
                    i=fr.index(v); sh=[1]*k; sh[i]=3; sh[k-1]=3
                    # M_ indexed [colour of u, colour of v] -> need [v-axis i, u-axis last]
                    vals=vals+M_.T.reshape(sh)
            fr.append(u)
            for i in reversed(range(len(fr))):
                if last[fr[i]]<=idx:
                    vals=(logsumexp(vals,axis=i) if T>0 else vals.min(axis=i)); fr.pop(i)
            if T>0:
                mx=vals.max(); vals=vals-mx; shift+=mx
            if idx+1==len(order) or key(xs[order[idx+1]])!=key(xs[u]):
                marks.append(shift+float(logsumexp(vals)) if T>0 else float(vals.min()))
        return marks
    def e(s,p,q,T=0,K=4):
        mk=s.run([0,p,q],T); a=(mk[-3]-mk[-3-K])/K
        return (a/s.N) if T==0 else (-T*a/s.N)
