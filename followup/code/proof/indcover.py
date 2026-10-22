import itertools, random, networkx as nx, numpy as np
from scipy.spatial import Delaunay
def has_indep_cover(G,F):
    # independent (in G) vertex set X covering all edges of F
    V=sorted({u for e in F for u in e})
    if not F: return True
    # backtracking
    adj={v:set(G[v]) for v in G}
    F=list(F)
    def rec(i,X):
        if i==len(F): return True
        u,v=F[i]
        if u in X or v in X: return rec(i+1,X)
        for w in (u,v):
            if not (adj[w]&X):
                X.add(w)
                if rec(i+1,X): return True
                X.remove(w)
        return False
    return rec(0,set())
random.seed(3); np.random.seed(3)
tot=0; allopt_fail=0; someopt_fail=0; ex=None
for trial in range(1500):
    n=random.randint(6,10)
    pts=np.random.rand(n,2); T=Delaunay(pts); G=nx.Graph(); G.add_nodes_from(range(n))
    keep=random.uniform(0.6,1.0)
    for s in T.simplices:
        for a in range(3):
            if random.random()<keep: G.add_edge(int(s[a]),int(s[(a+1)%3]))
    if not nx.is_connected(G): continue
    E=list(G.edges())
    # 3-colourable?
    ok3=False
    for col in itertools.product(range(3),repeat=n):
        if all(col[u]!=col[v] for u,v in E): ok3=True; break
    if not ok3: continue
    best=None; opts=[]
    for col in itertools.product(range(2),repeat=n):
        F=[(u,v) for u,v in E if col[u]==col[v]]
        b=len(F)
        if best is None or b<best: best=b; opts=[F]
        elif b==best: opts.append(F)
    tot+=1
    res=[has_indep_cover(G,F) for F in opts]
    if not any(res):
        allopt_fail+=1
        if ex is None: ex=(n,sorted(E),opts[0])
    elif not all(res): someopt_fail+=1
print("planar 3-colourable graphs:",tot)
print("  all min-frustration 2-colourings lack an independent cover:",allopt_fail)
print("  some (not all) lack one:",someopt_fail)
if ex: print("  example:",ex)
