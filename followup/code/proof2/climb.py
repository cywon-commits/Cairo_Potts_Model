import sys, random, networkx as nx, json
from ortools.sat.python import cp_model
def solve(m,t=10):
    s=cp_model.CpSolver(); s.parameters.num_workers=1; s.parameters.max_time_in_seconds=t
    st=s.Solve(m); return s.ObjectiveValue() if st==cp_model.OPTIMAL else None
def ioct(G):
    m=cp_model.CpModel(); x={(v,c):m.NewBoolVar('') for v in G for c in range(3)}
    for v in G: m.AddExactlyOne(x[v,c] for c in range(3))
    for u,v in G.edges():
        for c in range(3): m.AddBoolOr([x[u,c].Not(),x[v,c].Not()])
    m.Minimize(sum(x[v,2] for v in G)); return solve(m)
def eb(G):
    m=cp_model.CpModel(); y={v:m.NewBoolVar('') for v in G}; z=[]
    for u,v in G.edges():
        b=m.NewBoolVar(''); m.AddBoolOr([b,y[u],y[v]]); m.AddBoolOr([b,y[u].Not(),y[v].Not()]); z.append(b)
    m.Minimize(sum(z)); return solve(m)
mode=sys.argv[1]; seed=int(sys.argv[2]); n=int(sys.argv[3]); rnd=random.Random(seed)
def ok(G):
    if not nx.is_connected(G): return False
    if mode=='trianglefree' and sum(nx.triangles(G).values())>0: return False
    if mode=='planar4' and (max(d for _,d in G.degree())>4 or not nx.check_planarity(G)[0]): return False
    return True
def score(G):
    i=ioct(G)
    if i is None: return None
    e=eb(G); return i-e, i, e
G=nx.cycle_graph(n)
while True:
    H=G.copy(); u,v=rnd.sample(range(n),2)
    if H.has_edge(u,v): continue
    H.add_edge(u,v)
    if ok(H) and score(H) is not None: G=H; break
best=score(G); bestG=G.copy(); cur=best
for it in range(1500):
    H=G.copy()
    for _ in range(rnd.randint(1,2)):
        u,v=rnd.sample(range(n),2)
        if H.has_edge(u,v): H.remove_edge(u,v)
        else: H.add_edge(u,v)
    if not ok(H): continue
    s=score(H)
    if s is None: continue
    if s[0]>=cur[0] or rnd.random()<0.03:
        G=H; cur=s
        if s[0]>best[0]: best=s; bestG=H.copy(); print(it,'best',s,flush=True)
print('FINAL',mode,seed,n,best,sorted(bestG.edges()),flush=True)
