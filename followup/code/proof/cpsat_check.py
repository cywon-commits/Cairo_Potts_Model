import sys, random, numpy as np, networkx as nx
from scipy.spatial import Delaunay
from ortools.sat.python import cp_model
SC=8
GRID=[(pp,qq) for pp in (0.25,0.5,0.75,1.0) for qq in (0.0,pp/2,pp)]
def solve(model):
    s=cp_model.CpSolver(); s.parameters.num_workers=1; s.parameters.max_time_in_seconds=20
    st=s.Solve(model); return s.ObjectiveValue() if st in (cp_model.OPTIMAL,) else None
def e3(G,pp,qq):
    m=cp_model.CpModel(); x={(v,c):m.NewBoolVar('') for v in G for c in range(3)}
    for v in G: m.AddExactlyOne(x[v,c] for c in range(3))
    for u,v in G.edges():
        for c in range(3): m.AddBoolOr([x[u,c].Not(),x[v,c].Not()])
    m.Minimize(sum(int(round(qq*SC))*x[v,1]+int(round(pp*SC))*x[v,2] for v in G)); return solve(m)
def e2(G,qq):
    m=cp_model.CpModel(); y={v:m.NewBoolVar('') for v in G}; z=[]
    for u,v in G.edges():
        b=m.NewBoolVar(''); m.Add(b>=1-y[u]-y[v]); m.Add(b>=y[u]+y[v]-1); z.append(b)
    m.Minimize(SC*sum(z)+int(round(qq*SC))*sum(y.values())); return solve(m)
def gen(rng,n,kind):
    if kind=='planar':
        pts=rng.random((n,2)); T=Delaunay(pts); G=nx.Graph(); G.add_nodes_from(range(n)); keep=rng.uniform(0.55,0.95)
        for s in T.simplices:
            for a in range(3):
                if rng.random()<keep: G.add_edge(int(s[a]),int(s[(a+1)%3]))
    elif kind=='gnp': G=nx.gnp_random_graph(n,rng.uniform(0.15,0.35),seed=int(rng.integers(1e9)))
    else: G=nx.random_regular_graph(3 if n%2==0 else 4,n,seed=int(rng.integers(1e9)))
    return G
def main():
    seed=int(sys.argv[1])
    if __name__=="__main__": seed=int(sys.argv[1]); N=int(sys.argv[2]); rng=np.random.default_rng(seed)
    tested=0; viol=0; worst=0.0; ex=None; kinds={'planar':0,'gnp':0,'reg':0}
    while tested<N:
        n=int(rng.integers(11,21)); kind=['planar','planar','gnp','reg'][int(rng.integers(4))]
        G=gen(rng,n,kind)
        if not nx.is_connected(G): continue
        if e3(G,1.0,0.5) is None: continue      # not 3-colourable
        tested+=1; kinds[kind]+=1
        for pp,qq in GRID:
            a=e3(G,pp,qq); b=e2(G,qq)
            if a>b+1e-9:
                viol+=1; d=(a-b)/SC
                if d>worst: worst=d; ex=(kind,n,sorted(G.edges()),pp,qq)
    print(f"seed {seed}: tested {tested} graphs {kinds}, n in [11,20], {len(GRID)} cost points each; violations {viol}; worst {worst}; ex {ex}",flush=True)
if __name__=="__main__": main()
