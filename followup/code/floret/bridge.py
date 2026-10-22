import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,json,time; sys.path.insert(0,ROOT+"/floret"); sys.path.insert(1,ROOT+"/fam")
import numpy as np
from floret import build_floret, a1, a2
from tm_eig import marks
PER=7*math.sqrt(3)/2/math.sqrt(7)
def build_bridged(W,L,rule):
    n,E,xs,typ,pos=build_floret(W,L)
    adj=[set() for _ in range(n)]
    for a,b in E: adj[a].add(b); adj[b].add(a)
    E=set(E); xs=list(xs)
    nn=n
    for h in range(n):
        if typ[h]!='H': continue
        r=pos[h]; M=np.array([a1,a2]).T; i,j=np.round(np.linalg.solve(M,r)).astype(int)
        m=(3*i+j)//7; k=(-i+2*j)//7
        cands=[]
        for p in adj[h]:
            for x in adj[p]:
                if x==h or typ[x]!='T': continue
                if any(typ[y]=='H' for y in adj[x]): continue
                d=pos[x]-r; cands.append((math.atan2(d[1],d[0]),x))
        cands=sorted(set(cands))
        if not cands: continue
        idx=rule(m,k,len(cands))
        if idx is None: continue
        x=cands[idx%len(cands)][1]
        xs.append(xs[h]*0.5+xs[x]*0.5); E.add((h,nn)); E.add((x,nn)); nn+=1
    return nn,sorted(E),np.array(xs)
class End:
    def __init__(s,graph,spc,period,K=4): s.n,s.E,s.xs=graph; s.spc=spc; s.per=period; s.K=K; s.bad=0
    def e(s,p,q):
        m=marks(s.n,s.E,s.xs,[0,p,q],lambda x: math.floor((x+1e-6)/s.per)); K=s.K
        a=(m[-3]-m[-3-K])/K; b=(m[-3-K]-m[-3-2*K])/K
        if abs(a-b)>1e-9: s.bad+=1
        return a/s.spc
def inv(B):
    h=1e-3
    fe=(B.e(0,0.1+h)-B.e(0,0.1-h))/(2*h); fm=(B.e(0.1+h,0.3)-B.e(0.1-h,0.3))/(2*h)
    bm=B.e(0,60.0); g=(B.e(60,0.1+h)-B.e(60,0.1-h))/(2*h); return fe,fm,bm,g
RULES={"k0":lambda m,k,L:0,"k1":lambda m,k,L:1,"alt_n":lambda m,k,L:(k%2)*(L//2),"alt_m":lambda m,k,L:(m%2)*(L//2),
       "alt_n1":lambda m,k,L:(k%2),"alt_mn":lambda m,k,L:((m+k)%2)*(L//2)}
if __name__=="__main__":
    W=2; M=24
    for nm,rule in RULES.items():
        g_=build_bridged(W,M*2*PER+0.01,rule)
        n=g_[0]; xs=g_[2]; mid=(M//2)*2*PER
        spc=int(((xs>=mid)&(xs<mid+2*PER)).sum())
        B=End(g_,spc,2*PER)
        fe,fm,bm,g=inv(B)
        print(f"{nm:7s} sites/2-period={spc} f_exp={fe*10:.3f}/10 f_mid={fm*10:.3f}/10 b_min={bm*10:.3f}/10 g={g*10:.3f}/10 | c_c(avg)={bm/fe:.3f} s={(g-fm)/fe:+.3f}",flush=True)
