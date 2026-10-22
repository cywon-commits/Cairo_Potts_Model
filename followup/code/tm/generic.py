import numpy as np
def solve_cyl(n,TE,xs,p,q,prep=None):
    """min-plus DP over sites ordered by xs; returns dict x->cumulative min energy"""
    order=sorted(range(n),key=lambda u:(round(xs[u],6),u))
    rank={u:k for k,u in enumerate(order)}
    adj=[[] for _ in range(n)]
    for a,b in TE: adj[a].append(b); adj[b].append(a)
    D=np.array([0.0,p,q]); EQ=(np.arange(3)[:,None]==np.arange(3)[None,:]).astype(float)
    fr=[]; vals=np.zeros(()); marks={}
    for idx,u in enumerate(order):
        nb=[v for v in adj[u] if v in fr]
        vals=vals[...,None]+np.zeros(3); k=vals.ndim
        for v in nb:
            i=fr.index(v); sh=[1]*k; sh[i]=3; sh[k-1]=3; vals=vals+EQ.reshape(sh)
        sh=[1]*k; sh[k-1]=3; vals=vals+D.reshape(sh)
        fr.append(u)
        while True:
            drop=None
            for i,v in enumerate(fr):
                if all(rank[w]<=rank[u] for w in adj[v]): drop=i;break
            if drop is None: break
            vals=vals.min(axis=drop); fr.pop(drop)
        if idx+1<len(order) and round(xs[order[idx+1]],6)!=round(xs[u],6):
            marks[round(xs[u],6)]=float(vals.min())
    marks[round(xs[order[-1]],6)]=float(vals.min())
    return marks

def cairo_cyl(W,M):
    """Cairo pentagonal lattice: periodic in y (W cells), open in x (M cells)."""
    sites={}; pos=[]; xs=[]
    def add(key,xy):
        sites[key]=len(pos); pos.append(xy); xs.append(xy[0]); return sites[key]
    for i in range(M):
        for j in range(W):
            add(('A',i,j),(i+0.0,j+0.0)); add(('B',i,j),(i+0.5,j+0.5))
            add(('Vp',i,j),(i+0.5,j+0.15)); add(('Vm',i,j),(i+0.5,j-0.15))
            add(('Hp',i,j),(i+0.15,j+0.5)); add(('Hm',i,j),(i-0.15,j+0.5))
    E=set()
    def link(k1,k2):
        if k1 in sites and k2 in sites:
            a,b=sites[k1],sites[k2]; E.add((min(a,b),max(a,b)))
    for i in range(M):
        for j in range(W):
            jp=(j+1)%W; jm=(j-1)%W
            link(('Vp',i,j),('Vm',i,j)); link(('Hp',i,j),('Hm',i,j))
            link(('Vp',i,j),('B',i,j));  link(('Vp',i,j),('A',i,j))
            link(('Vm',i,j),('B',i,jm)); link(('Vm',i,j),('A',i+1,j))
            link(('Hp',i,j),('B',i,j));  link(('Hp',i,j),('A',i,jp))
            link(('Hm',i,j),('B',i-1,j));link(('Hm',i,j),('A',i,j))
    return len(pos),sorted(E),np.array(xs),sites

if __name__=="__main__":
    from collections import Counter
    for W in [2,3]:
        n,E,xs,sites=cairo_cyl(W,5)
        deg=[0]*n
        for a,b in E: deg[a]+=1; deg[b]+=1
        print(f"W={W} N={n} E={len(E)} degdist={Counter(deg)}")
    # independent-OCT check on a torus-like big patch (bulk): use DP at p=q small to read composition
