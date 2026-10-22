import numpy as np, itertools, random
from math import atan2, sqrt

class Lat:
    def __init__(s,pos,box): s.pos=np.array(pos,float); s.box=np.array(box,float); s.E=set()
    def disp(s,i,j):
        d=s.pos[j]-s.pos[i]
        # reduce by lattice vectors a1,a2 (box = 2x2 matrix rows a1,a2)
        M=s.box.T; inv=np.linalg.inv(M)
        f=inv@d; f=f-np.round(f); return M@f
    def auto_bonds(s,target,tol=1e-6):
        n=len(s.pos)
        for i in range(n):
            for j in range(i+1,n):
                if abs(np.linalg.norm(s.disp(i,j))-target)<tol: s.E.add((i,j))
    def edges(s): return sorted(s.E)

def torus_tile(base,L,a1,a2):
    pos=[];
    for i in range(L):
        for j in range(L):
            for b in base: pos.append(np.array(b)+i*np.array(a1)+j*np.array(a2))
    return pos,[np.array(a1)*L,np.array(a2)*L]

def honeycomb(L):
    a=1/sqrt(3); a1=(1,0); a2=(0.5,sqrt(3)/2)
    base=[(0,0),(0,a)]
    pos,box=torus_tile(base,L,a1,a2); lat=Lat(pos,box); lat.auto_bonds(a); return lat

def dice(L):
    a=1/sqrt(3); a1=(1,0); a2=(0.5,sqrt(3)/2)
    base=[(0,0),(0,a),(0.5,a/2)]   # A, B, hub C
    pos,box=torus_tile(base,L,a1,a2); lat=Lat(pos,box)
    n=len(pos)
    for i in range(n):
        for j in range(i+1,n):
            ti,tj=i%3,j%3
            if (ti==2)!=(tj==2):
                if abs(np.linalg.norm(lat.disp(i,j))-a)<1e-6: lat.E.add((i,j))
    return lat

def square(L):
    pos,box=torus_tile([(0,0)],L,(1,0),(0,1)); lat=Lat(pos,box); lat.auto_bonds(1.0); return lat

def degrees(lat):
    n=len(lat.pos); g=[[] for _ in range(n)]
    for a,b in lat.edges(): g[a].append(b); g[b].append(a)
    return g

def truncate(lat):
    g=degrees(lat); n=len(lat.pos)
    vid={}; verts=[]
    for v in range(n):
        order=sorted(g[v], key=lambda w: atan2(*lat.disp(v,w)[::-1]))
        for w in order: vid[(v,w)]=len(verts); verts.append((v,w))
    E=set()
    for v in range(n):
        order=sorted(g[v], key=lambda w: atan2(*lat.disp(v,w)[::-1]))
        k=len(order)
        for t in range(k):
            a,b=vid[(v,order[t])],vid[(v,order[(t+1)%k])]; E.add((min(a,b),max(a,b)))
        polys=None
    polygons=[[vid[(v,w)] for w in sorted(g[v], key=lambda w: atan2(*lat.disp(v,w)[::-1]))] for v in range(n)]
    for a,b in lat.edges():
        x,y=vid[(a,b)],vid[(b,a)]; E.add((min(x,y),max(x,y)))
    return len(verts), sorted(E), polygons

def adj(n,E):
    g=[set() for _ in range(n)]
    for a,b in E: g[a].add(b); g[b].add(a)
    return g

def bip(n,g,S):
    col={}; comps=[]
    for s in range(n):
        if s in S or s in col: continue
        col[s]=0; st=[s]; cc=[s]
        while st:
            u=st.pop()
            for w in g[u]:
                if w in S: continue
                if w not in col: col[w]=1-col[u]; st.append(w); cc.append(w)
                elif col[w]==col[u]: return None,None
        a=sum(1 for v in cc if col[v]==0); comps.append((a,len(cc)-a))
    return col,comps

def analyze(name,lat,tries=4000):
    n,E,polys=truncate(lat); g=adj(n,E)
    deg=set(len(x) for x in g)
    odd=[p for p in polys if len(p)%2==1]
    lb=len(odd)
    best=None
    rng=random.Random(0)
    for t in range(tries):
        S=set(rng.choice(p) for p in odd)
        if len(S)<lb: continue
        if any(b in S for a,b in E if a in S): continue
        col,comps=bip(n,g,S)
        if col is not None: best=(S,comps); break
    print(f"--- {name}: N={n}, degrees={deg}, odd faces={lb}, lower bound f_exp>={lb/n if n else 0:.4f}")
    if best:
        S,comps=best
        fmid=sum(min(a,b) for a,b in comps)/n
        print(f"    independent OCT found: |S|={len(S)}  f_exp={len(S)/n:.4f}  f_mid={fmid:.4f}  (eta=1, optimal)")
    elif lb==0:
        col,comps=bip(n,g,set())
        print(f"    bipartite -> f_exp=0, f_mid={sum(min(a,b) for a,b in comps)/n:.4f}")
    else:
        print("    no independent optimal OCT found in search")

for L in []:
    analyze(f"truncated honeycomb (3.12.12) L={L}", honeycomb(L))
    analyze(f"truncated square  (4.8.8)    L={L}", square(L))
    analyze(f"truncated dice                L={L}", dice(L))
