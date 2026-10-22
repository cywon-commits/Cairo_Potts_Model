import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/fam")
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq
from lib import cylinder, a2
from lattices import honey_points, dice_rule
PER=math.sqrt(3)/2
# dice lattice cylinder for the effective Ising model (spins on H rims and K centres)
def dice(W,M):
    n,E,xs,typ,pos,disp=cylinder(honey_points(R=M+14,hubs=lambda i,j:True),W*a2,M*PER+0.01,dice_rule)
    return n,E,xs
def logZ_per_bond(n,E,xs,K,W):
    order=sorted(range(n),key=lambda u:(round(xs[u],6),u)); rank={u:k for k,u in enumerate(order)}
    adj=[[] for _ in range(n)]
    for a,b in E: adj[a].append(b); adj[b].append(a)
    last=[max([rank[w] for w in adj[v]]+[rank[v]]) for v in range(n)]
    J=np.array([[K,-K],[-K,K]]); fr=[]; vals=np.zeros(()); marks=[]; shift=0.0
    key=lambda x: math.floor((x+1e-6)/PER)
    for idx,u in enumerate(order):
        vals=vals[...,None]+np.zeros(2); k=vals.ndim
        for v in adj[u]:
            if v in fr:
                i=fr.index(v); sh=[1]*k; sh[i]=2; sh[k-1]=2; vals=vals+J.reshape(sh)
        fr.append(u)
        for i in reversed(range(len(fr))):
            if last[fr[i]]<=idx: vals=logsumexp(vals,axis=i); fr.pop(i)
        mx=vals.max(); vals=vals-mx; shift+=mx
        if idx+1==len(order) or key(xs[order[idx+1]])!=key(xs[u]): marks.append(shift+float(logsumexp(vals)))
    Kp=4; return (marks[-3]-marks[-3-Kp])/Kp/(6*W)   # 6 dice bonds per cell
def corr(K,g,h=1e-4):  # <sigma sigma'> = d(lnZ per bond)/dK
    return (logZ_per_bond(*g,K+h,W)-logZ_per_bond(*g,K-h,W))/(2*h)
W=3; g=dice(W,30)
def healed_fraction(x):
    K=0.5*math.log((2+x)/(1+x)); c=corr(K,g); p=(1+c)/2
    return p*2/(2+x)+(1-p)/(1+x), p, K
xs_=brentq(lambda x: healed_fraction(x)[0]-0.5, 1.0, 2.0, xtol=1e-10)
h,p,K=healed_fraction(xs_)
print(f"effective dice-Ising: midpoint x* = {xs_:.5f}  ->  shift (t*-1)/T = ln x* = {math.log(xs_):.5f}")
print(f"   at x*: K_eff = {K:.5f}, p_eq = {p:.5f}  (uncorrelated would give ln sqrt2 = {math.log(math.sqrt(2)):.5f})")
print("measured from full model: 0.0165/0.04=0.4125, 0.0329/0.08=0.4113, 0.0494/0.12=0.4117")
# full n(t) curve at T=0.08 vs measured
meas={0.85:8.467,0.90:8.067,0.95:7.448,0.98:6.970,1.00:6.616,1.02:6.244,1.04:5.866,1.06:5.493,1.10:4.805,1.15:4.122}
print("\nT=0.08   t    n*16 predicted (3+6h)   n*16 measured   independent-2-level")
for t,m in meas.items():
    x=math.exp((t-1)/0.08); hh=healed_fraction(x)[0]
    print(f"        {t:.2f}      {3+6*hh:7.3f}             {m:7.3f}          {9-6/(1+math.exp(-(t-1)/0.08)):7.3f}")
