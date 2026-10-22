from gadgets import *
import sys
def trideco(L,m):
    G=nx.Graph(); H=lambda i,j:('h',i%L,j%L)
    for i in range(L):
        for j in range(L):
            for di,dj in [(1,0),(0,1),(1,-1)]:
                u,v=H(i,j),H(i+di,j+dj); G.add_edge(u,v)
                for k in range(m):
                    s=('s',i,j,di,dj,k); G.add_edge(s,u); G.add_edge(s,v)
    return G
def eall(G,p,q,SC=20):
    m=cp_model.CpModel(); x={(v,c):m.NewBoolVar('') for v in G for c in range(3)}
    for v in G: m.AddExactlyOne(x[v,c] for c in range(3))
    z=[]
    for u,v in G.edges():
        for c in range(3):
            b=m.NewBoolVar(''); m.AddBoolOr([b,x[u,c].Not(),x[v,c].Not()]); z.append(b)
    m.Minimize(SC*sum(z)+sum(int(round(q*SC))*x[v,1]+int(round(p*SC))*x[v,2] for v in G))
    val,s=solve(m); n3=sum(s.Value(x[v,2]) for v in G); b=sum(s.Value(t) for t in z); return val/SC,n3,b
L=6
for mm in [1,2,3,4,6]:
    G=trideco(L,mm); nh=L*L
    print(f"m={mm}: per hub site: ioct={ioct(G)/nh:.3f} (pred {1/3+mm:.3f})  eb={eb(G)/nh:.3f} (pred <=3)  predicted p'_c <= {3/(1/3+mm):.3f}",flush=True)
    for p in [0.3,0.5,0.7,0.9,1.0]:
        e,n3,b=eall(G,p,0.0); print(f"     p'={p}: ground e/hub={e/nh:.3f}  n3/hub={n3/nh:.3f}  b/hub={b/nh:.3f}  three-colour e3/hub={p*(1/3+mm):.3f}",flush=True)
