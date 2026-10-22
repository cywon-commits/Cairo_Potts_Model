import numpy as np, math
from math import sqrt, atan2
a1=np.array([1.0,0.0]); a2=np.array([0.5,sqrt(3)/2]); A=1/sqrt(3)

def cylinder(points, P, L, bondrule):
    """points: list of (type, xy) generated over a big region.
    P: periodic vector. keep points with perpendicular coord in [0,L)."""
    P=np.array(P,float); u=np.array([-P[1],P[0]])/np.linalg.norm(P)
    keep={}; 
    for typ,r in points:
        t=float(r@u)
        if not (-1e-9<=t<L-1e-9): continue
        s=float(r@P/(P@P)); s-=math.floor(s+1e-9)
        key=(round(s,6)%1.0,round(t,6))
        keep[key]=(typ,r - math.floor(float(r@P/(P@P))+1e-9)*P)
    items=list(keep.values())
    typ=[x[0] for x in items]; pos=np.array([x[1] for x in items])
    def disp(i,j):
        d=pos[j]-pos[i]; k=round(float(d@P/(P@P))); return d-k*P
    n=len(pos); E=set()
    for i in range(n):
        for j in range(i+1,n):
            d=disp(i,j)
            if np.linalg.norm(d)<1.2*A+1e-9:
                if bondrule(typ[i],typ[j],d): E.add((i,j))
    xs=pos@u
    return n,sorted(E),xs,typ,pos,disp

def truncate(n,E,disp,xs):
    g=[[] for _ in range(n)]
    for a,b in E: g[a].append(b); g[b].append(a)
    vid={}; parent=[]
    order={}
    for v in range(n):
        o=sorted(g[v],key=lambda w:atan2(disp(v,w)[1],disp(v,w)[0])); order[v]=o
        for w in o: vid[(v,w)]=len(parent); parent.append(v)
    TE=set()
    for v in range(n):
        o=order[v]; k=len(o)
        if k==2: a,b=vid[(v,o[0])],vid[(v,o[1])]; TE.add((min(a,b),max(a,b)))
        elif k>2:
            for t in range(k):
                a,b=vid[(v,o[t])],vid[(v,o[(t+1)%k])]; TE.add((min(a,b),max(a,b)))
    for a,b in E:
        x,y=vid[(a,b)],vid[(b,a)]; TE.add((min(x,y),max(x,y)))
    nx=np.array([xs[p] for p in parent])
    # small offset along the bond so ordering is well defined
    for (v,w),k in vid.items(): nx[k]+=1e-4*np.sign(disp(v,w)@np.array([1,0]) if False else 0)
    return len(parent),sorted(TE),nx

def ground_energy(n,E,xs,D):
    order=sorted(range(n),key=lambda u:(round(xs[u],6),u))
    rank={u:k for k,u in enumerate(order)}
    adj=[[] for _ in range(n)]
    for a,b in E: adj[a].append(b); adj[b].append(a)
    last=[max([rank[w] for w in adj[v]]+[rank[v]]) for v in range(n)]
    D=np.asarray(D,float); EQ=np.eye(3)
    fr=[]; vals=np.zeros(()); mx=0
    for idx,u in enumerate(order):
        vals=vals[...,None]+D.reshape([1]*vals.ndim+[3]); k=vals.ndim
        for v in adj[u]:
            if v in fr:
                i=fr.index(v); sh=[1]*k; sh[i]=3; sh[k-1]=3; vals=vals+EQ.reshape(sh)
        fr.append(u)
        for i in reversed(range(len(fr))):
            if last[fr[i]]<=idx: vals=vals.min(axis=i); fr.pop(i)
        mx=max(mx,len(fr))
    return float(vals), mx

def frontier(n,E,xs):
    order=sorted(range(n),key=lambda u:(round(xs[u],6),u)); rank={u:k for k,u in enumerate(order)}
    adj=[[] for _ in range(n)]
    for a,b in E: adj[a].append(b); adj[b].append(a)
    last=[max([rank[w] for w in adj[v]]+[rank[v]]) for v in range(n)]
    fr=set(); mx=0
    for idx,u in enumerate(order):
        fr.add(u); fr={v for v in fr if last[v]>idx}; mx=max(mx,len(fr)+1)
    return mx

class Bulk:
    """per-site bulk energy from two lengths L1<L2 (boundary cancels)."""
    def __init__(s,build,L1,L2):
        s.g1=build(L1); s.g2=build(L2)
    def e(s,p,q):
        e1,_=ground_energy(s.g1[0],s.g1[1],s.g1[2],[0,p,q]); e2,_=ground_energy(s.g2[0],s.g2[1],s.g2[2],[0,p,q])
        return (e2-e1)/(s.g2[0]-s.g1[0])
