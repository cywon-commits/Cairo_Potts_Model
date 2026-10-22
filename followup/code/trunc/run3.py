import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys; sys.path.insert(0,ROOT+"/trunc")
from trunc import *
def uniform_test(name,lat,L,ncell):
    n,E,polys=truncate(lat); g=adj(n,E)
    odd=[(v,p) for v,p in enumerate(polys) if len(p)%2==1]
    types=sorted(set(v%ncell for v,p in odd))
    print(f"{name} L={L}: N={n}, odd faces={len(odd)}, sublattice types {types}")
    best=None
    for choice in itertools.product(range(3),repeat=len(types)):
        rule={t:c for t,c in zip(types,choice)}
        S=set(p[rule[v%ncell]] for v,p in odd)
        if len(S)!=len(odd): continue
        if any(b in S for a,b in E if a in S): continue
        col,comps=bip(n,g,S)
        if col is not None:
            fmid=sum(min(a,b) for a,b in comps)/n
            print(f"   rule {rule}: f_exp={len(S)/n:.4f}={len(S)}/{n}  f_mid={fmid:.4f}  OK (eta=1)")
            best=choice; break
    if best is None: print("   no uniform rule works")
for L in [2,3,4,5]:
    uniform_test("truncated dice", dice(L), L, 3)
for L in [4,5]:
    uniform_test("truncated honeycomb", honeycomb(L), L, 2)
