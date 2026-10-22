import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys; sys.path.insert(0,ROOT+"/trunc")
from trunc import *
import random, math

def sa(n,g,D,restarts=12,sweeps=4000):
    best=None
    for r in range(restarts):
        rng=random.Random(r*7919+1)
        s=[rng.randrange(3) for _ in range(n)]
        def site_e(u,c): return sum(1 for w in g[u] if s[w]==c)+D[c]
        for t in range(sweeps):
            T=3.0*(0.002/3.0)**(t/(sweeps-1))
            for _ in range(n):
                u=rng.randrange(n); c=rng.randrange(3)
                if c==s[u]: continue
                dE=site_e(u,c)-site_e(u,s[u])
                if dE<=0 or rng.random()<math.exp(-dE/T): s[u]=c
        E=sum(1 for a in range(n) for w in g[a] if s[w]==s[a])/2+sum(D[c] for c in s)
        if best is None or E<best[0]: best=(E,list(s))
    return best

def scan(name,lat):
    n,E,polys=truncate(lat); g=adj(n,E)
    print(f"\n{name}  N={n}")
    print("   p     q    E/N(SA)   n3/N   n2/N  | Potts pred  2col pred")
    for p,q in [(1,0.6),(1,0.9),(1,1.1),(1,1.4),(2,0.9),(2,1.1),(0.5,0.9),(0.5,1.1)]:
        e,s=sa(n,g,[0.0,p,q])
        n3=sum(1 for c in s if c==2)/n; n2=sum(1 for c in s if c==1)/n
        fe,fm=(1/3,1/3) if 'honey' in name else (1/6,5/12)
        pot=p*fm+q*fe
        two=(1/3)*(1 if 'honey' in name else None) if 'honey' in name else None
        twoc= (1/3+p/3) if 'honey' in name else float('nan')
        print(f" {p:4.1f} {q:5.2f}  {e/n:7.4f}  {n3:.4f} {n2:.4f} |  {pot:.4f}     {twoc:.4f}")

scan("truncated honeycomb", honeycomb(3))
scan("truncated dice", dice(2))
