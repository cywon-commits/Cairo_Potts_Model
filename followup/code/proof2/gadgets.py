import networkx as nx
from ortools.sat.python import cp_model
def solve(m):
    s=cp_model.CpSolver(); s.parameters.num_workers=8; s.parameters.max_time_in_seconds=120
    st=s.Solve(m); return (s.ObjectiveValue() if st==cp_model.OPTIMAL else None), s
def ioct(G):   # min |colour 3| over proper 3-colourings  (= E3 at p=1,q=0)
    m=cp_model.CpModel(); x={(v,c):m.NewBoolVar('') for v in G for c in range(3)}
    for v in G: m.AddExactlyOne(x[v,c] for c in range(3))
    for u,v in G.edges():
        for c in range(3): m.AddBoolOr([x[u,c].Not(),x[v,c].Not()])
    m.Minimize(sum(x[v,2] for v in G)); return solve(m)[0]
def eb(G):     # min frustrated bonds over 2-colourings (= E2 at q=0)
    m=cp_model.CpModel(); y={v:m.NewBoolVar('') for v in G}; z=[]
    for u,v in G.edges():
        b=m.NewBoolVar(''); m.Add(b>=y[u]-y[v]).OnlyEnforceIf([]) if False else None
        m.AddBoolOr([b,y[u],y[v]]); m.AddBoolOr([b,y[u].Not(),y[v].Not()]); z.append(b)
    m.Minimize(sum(z)); return solve(m)[0]
def Y(s):
    G=nx.Graph(); n=3*s
    for P in 'abcd':
        for i in range(1,n): G.add_edge((P,i),(P,i+1))
    for i in range(1,n+1): G.add_edge(('a',i),('b',i)); G.add_edge(('c',i),('d',i))
    for i in range(1,n): G.add_edge(('a',i),('c',i+1)); G.add_edge(('d',i),('b',i+1))
    G.add_edge('x','y'); G.add_edge('x',('a',1)); G.add_edge('x',('d',1))
    for P in 'abcd': G.add_edge('y',(P,1))
    return G
def Z(s):
    G=nx.Graph(); G.add_edge('a','b')
    for i in range(s):
        p=[(i,k) for k in range(4)]; nx.add_path(G,p)
        for h in 'ab': G.add_edge(h,p[0]); G.add_edge(h,p[3])
    return G
def Q(s):
    G=nx.Graph(); A=[('A',i) for i in range(s)]; B=[('B',i) for i in range(s)]; C=[('C',i) for i in range(s)]
    a,b,c=A[0],B[0],C[0]; G.add_nodes_from(A+B+C)
    for v in B+C: G.add_edge(a,v)
    for v in A+C:
        if v!=a: G.add_edge(b,v)
    G.add_edge(a,b)
    for v in A+B: G.add_edge(c,v)
    return G
if __name__=="__main__":
    for name,f in [('Y',Y),('Z',Z),('Q',Q)]:
        for s in [1,2,3,4,6]:
            G=f(s); i=ioct(G); e=eb(G)
            print(f"{name}_{s}: n={G.number_of_nodes()} maxdeg={max(d for _,d in G.degree())} planar={nx.check_planarity(G)[0]}  ioct={i}  eb={e}  axis threshold p'_c=eb/ioct={e/i if i else None:.3f}",flush=True)
