from gadgets import *
def Yper(k,L):
    N=k*L; G=nx.Graph(); nx_=lambda i:(i-1)%N+1
    for P in 'abcd':
        for i in range(1,N+1): G.add_edge((P,i),(P,nx_(i+1)))
    for i in range(1,N+1): G.add_edge(('a',i),('b',i)); G.add_edge(('c',i),('d',i)); G.add_edge(('a',i),('c',nx_(i+1))); G.add_edge(('d',i),('b',nx_(i+1)))
    for j in range(k):
        s=1+j*L; x=('x',j); y=('y',j); G.add_edge(x,y); G.add_edge(x,('a',s)); G.add_edge(x,('d',s))
        for P in 'abcd': G.add_edge(y,(P,s))
    return G
for k in [1,2,3]:
    for L in [3,6,9,12,18]:
        G=Yper(k,L); i=ioct(G); e=eb(G)
        print(f"k={k} L={L}: n={G.number_of_nodes()} ioct={i} eb={e} ratio={'not 3-colourable' if i is None else round(e/i,3)}",flush=True)
