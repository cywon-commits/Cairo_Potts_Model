import math, json, sys, numpy as np
INF=1e9
def edge_tables(cost,m,proper):
    W=np.zeros((3,3))
    for a in range(3):
        for b in range(3):
            if proper:
                if a==b: W[a,b]=INF
                else: W[a,b]=m*cost[3-a-b]
            else:
                W[a,b]=(a==b)+m*min(cost[s]+(s==a)+(s==b) for s in range(3))
    return W
def energy(cost,m,proper,Wd=6,R=14,K=3):
    Wt=edge_tables(cost,m,proper); site=np.array(cost,float)
    order=[(r,c) for r in range(R) for c in range(Wd)]; idx={s:k for k,s in enumerate(order)}
    nb={s:[] for s in order}
    for (r,c) in order:
        for dr,dc in [(0,1),(1,0),(1,-1)]:
            t=(r+dr,(c+dc)%Wd)
            if t in idx: nb[(r,c)].append(t); nb[t].append((r,c))
    last={s:max([idx[t] for t in nb[s]]+[idx[s]]) for s in order}
    fr=[]; vals=np.zeros(()); shift=0.0; marks=[]
    for k,s in enumerate(order):
        vals=vals[...,None]+site.reshape([1]*vals.ndim+[3]); nd=vals.ndim
        for t in nb[s]:
            if t in fr:
                i=fr.index(t); sh=[1]*nd; sh[i]=3; sh[nd-1]=3; vals=vals+Wt.reshape(sh)
        fr.append(s)
        for i in reversed(range(len(fr))):
            if last[fr[i]]<=k: vals=vals.min(axis=i); fr.pop(i)
        mn=vals.min(); vals=vals-mn; shift+=mn
        if s[1]==Wd-1: marks.append(shift+vals.min())
    return (marks[-2]-marks[-2-K])/(K*Wd)
def inside(p,q,m):
    c=(0.0,p,q); return energy(c,m,True)<=energy(c,m,False)+1e-7
if __name__=="__main__":
    m=int(sys.argv[1]); pts=[]
    for ang in range(0,360,10):
        th=math.radians(ang); lo,hi=0.0,4.5
        if inside(hi*math.cos(th),hi*math.sin(th),m): pts.append((ang,hi*math.cos(th),hi*math.sin(th))); print(ang,'>=4.5',flush=True); continue
        for _ in range(14):
            r=(lo+hi)/2
            if inside(r*math.cos(th),r*math.sin(th),m): lo=r
            else: hi=r
        pts.append((ang,lo*math.cos(th),lo*math.sin(th))); print(ang,round(lo*math.cos(th),4),round(lo*math.sin(th),4),flush=True)
    json.dump(pts,open(f'tri_region_m{m}.json','w'))
