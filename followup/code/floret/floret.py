import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/fam")
import numpy as np
from lib import ground_energy, frontier
from tm_eig import marks
a1=np.array([1.0,0.0]); a2=np.array([0.5,math.sqrt(3)/2])
def removed(i,j): return (3*i+j)%7==0
def build_floret(W,L,R=None):
    R=R or int(L*1.3)+12
    P=W*(2*a1+a2); Pn=P@P; u=np.array([-P[1],P[0]])/math.sqrt(Pn)
    def key(r):
        s=(r@P)/Pn; s-=math.floor(s+1e-9); return (round(s,6)%1.0, round(float(r@u),6))
    nodes={}; pos=[]; typ=[]
    def node(kind,r):
        t=float(r@u)
        if not (-1e-9<=t<L-1e-9): return None
        k=(kind,)+key(r)
        if k not in nodes: nodes[k]=len(pos); pos.append(r); typ.append(kind)
        return nodes[k]
    tri={}  # (i,j,'u'/'d') -> vertex triple
    def tri_verts(i,j,o): return [(i,j),(i+1,j),(i,j+1)] if o=='u' else [(i+1,j),(i,j+1),(i+1,j+1)]
    def cen(vs): return sum(i*a1+j*a2 for i,j in vs)/3
    edge_tris={}
    for i in range(-R,R):
        for j in range(-R,R):
            for o in 'ud':
                vs=tri_verts(i,j,o)
                for k in range(3):
                    e=tuple(sorted((vs[k],vs[(k+1)%3])))
                    edge_tris.setdefault(e,[]).append((i,j,o))
    E=set()
    for e,ts in edge_tris.items():
        if len(ts)!=2: continue
        A,B=ts
        va,vb=tri_verts(*A),tri_verts(*B)
        ra=[v for v in va if removed(*v)]; rb=[v for v in vb if removed(*v)]
        if ra and rb: continue
        if not ra and not rb:
            x=node('T',cen(va)); y=node('T',cen(vb))
        else:
            kept,other=(va,vb) if not ra else (vb,va)
            r=[v for v in other if removed(*v)][0]
            x=node('T',cen(kept)); y=node('H',r[0]*a1+r[1]*a2)
        if x is None or y is None or x==y: continue
        E.add((min(x,y),max(x,y)))
    xs=np.array([float(p@u) for p in pos])
    return len(pos),sorted(E),xs,typ,pos
if __name__=="__main__":
    from collections import Counter
    for W in [1,2]:
        n,E,xs,typ,_=build_floret(W,12.0)
        deg=Counter()
        for a,b in E: deg[a]+=1; deg[b]+=1
        inner=[v for v in range(n) if 2.5<xs[v]<9.5]
        print(f"W={W} N={n} types={Counter(typ)} bulk degrees={Counter((typ[v],deg[v]) for v in inner)} frontier={frontier(n,E,xs)}")
