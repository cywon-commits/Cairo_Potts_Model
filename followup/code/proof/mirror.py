# Claim M: on the mirror ray (costs 0, t, t) with t <= 1, a frustration-free colouring is optimal.
# equivalently  min_sigma [ b(sigma) + t * #non-1 ]  is attained with b=0, for t=1 (then for all t<=1).
import itertools, random, networkx as nx
def check(G):
    n=G.number_of_nodes(); E=list(G.edges())
    best_any=None; best_proper=None; witness=None
    for col in itertools.product(range(3),repeat=n):
        b=sum(1 for u,v in E if col[u]==col[v]); c=sum(1 for x in col if x!=0)
        e=b+c
        if best_any is None or e<best_any: best_any=e; witness=col
        if b==0 and (best_proper is None or c<best_proper): best_proper=c
    return best_any,best_proper,witness
random.seed(1); bad=0; tested=0
for trial in range(4000):
    n=random.randint(4,9)
    kind=random.random()
    if kind<0.4: G=nx.gnp_random_graph(n,random.uniform(0.25,0.7),seed=random.randint(0,10**9))
    elif kind<0.7:
        G=nx.random_regular_graph(3 if n%2==0 else 4,n,seed=random.randint(0,10**9))
    else:
        # random planar-ish: triangulation subgraph
        import numpy as np
        pts=np.random.rand(n,2); from scipy.spatial import Delaunay
        T=Delaunay(pts); G=nx.Graph(); G.add_nodes_from(range(n))
        for s in T.simplices:
            for a in range(3):
                if random.random()<0.75: G.add_edge(int(s[a]),int(s[(a+1)%3]))
    if not nx.is_connected(G): continue
    ba,bp,w=check(G)
    if bp is None: continue
    tested+=1
    if bp is None: continue
    if ba<bp:
        bad+=1
        if bad<=3: print("COUNTEREXAMPLE n=%d edges=%s  best_any=%s best_proper=%s witness=%s"%(n,sorted(G.edges()),ba,bp,w))
print("tested",tested,"graphs; violations of claim M at t=1:",bad)
