# Theorem 3 (two-colour version): for all 0<=q'<=p'<=1, min over proper colourings of (q' n_mid + p' n_exp)
#   <= min over two-colour states (colours cheap/mid only) of (b + q' n_mid).
import itertools, random, networkx as nx, numpy as np
from scipy.spatial import Delaunay
def colourings(n):
    return np.array(list(itertools.product(range(3),repeat=n)),dtype=np.int8)
CACHE={}
def test(G, grid):
    n=G.number_of_nodes(); E=np.array(list(G.edges()))
    if n not in CACHE: CACHE[n]=colourings(n)
    C=CACHE[n]
    mono=(C[:,E[:,0]]==C[:,E[:,1]]).sum(1)
    n1=(C==1).sum(1); n2=(C==2).sum(1)
    proper=mono==0
    if not proper.any(): return None
    two=(n2==0)                     # colours 0 (cheap) and 1 (mid) only
    worst=0.0; wit=None
    for pp,qq in grid:
        e3=(qq*n1[proper]+pp*n2[proper]).min()
        e2=(mono[two]+qq*n1[two]).min()
        if e2<e3-1e-9 and e3-e2>worst: worst=e3-e2; wit=(pp,qq)
    return worst,wit
grid=[(pp,qq) for pp in np.linspace(0.05,1.0,20) for qq in np.linspace(0,1.0,21) if qq<=pp]
random.seed(7); np.random.seed(7)
stats={"planar":[0,0],"nonplanar":[0,0]}
examples=[]
for trial in range(2500):
    n=random.randint(5,10)
    r=random.random()
    if r<0.5:
        pts=np.random.rand(n,2); T=Delaunay(pts); G=nx.Graph(); G.add_nodes_from(range(n))
        keep=random.uniform(0.55,1.0)
        for s in T.simplices:
            for a in range(3):
                if random.random()<keep: G.add_edge(int(s[a]),int(s[(a+1)%3]))
    elif r<0.75:
        G=nx.gnp_random_graph(n,random.uniform(0.25,0.6),seed=random.randint(0,10**9))
    else:
        d=3 if n%2==0 else 4
        G=nx.random_regular_graph(d,n,seed=random.randint(0,10**9))
    if not nx.is_connected(G): continue
    res=test(G,grid)
    if res is None: continue
    kind="planar" if nx.check_planarity(G)[0] else "nonplanar"
    stats[kind][0]+=1
    if res[0]>0:
        stats[kind][1]+=1
        if len(examples)<3: examples.append((kind,n,sorted(G.edges()),res))
print("tested (3-colourable):",{k:v[0] for k,v in stats.items()}," violations:",{k:v[1] for k,v in stats.items()})
for ex in examples: print("EXAMPLE",ex)
