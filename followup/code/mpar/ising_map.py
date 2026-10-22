import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/mpar"); sys.path.insert(1,ROOT+"/fam")
import numpy as np
from scipy.special import logsumexp
from decim import Dec, PER
def ising_lnZ_cell(B,m,x,K=4):
    # effective dice Ising on the same cylinder: H,K spins, bond weight (g+x)^m, g=2 same / 1 different; no K-K coupling
    L=m*np.log(np.array([[2+x,1+x],[1+x,2+x]]))
    n=B.n; xs=B.xs
    order=sorted(range(n),key=lambda u:(round(xs[u],6),u)); rank={u:k for k,u in enumerate(order)}
    adj=[[] for _ in range(n)]
    for a,b,kind in B.bonds:
        if kind=='u': adj[a].append(b); adj[b].append(a)
    last=[max([rank[w] for w in adj[v]]+[rank[v]]) for v in range(n)]
    fr=[]; vals=np.zeros(()); marks=[]; shift=0.0; key=lambda x_: math.floor((x_+1e-6)/PER)
    for idx,u in enumerate(order):
        vals=vals[...,None]+np.zeros(2); k=vals.ndim
        for v in adj[u]:
            if v in fr:
                i=fr.index(v); sh=[1]*k; sh[i]=2; sh[k-1]=2; vals=vals+L.reshape(sh)
        fr.append(u)
        for i in reversed(range(len(fr))):
            if last[fr[i]]<=idx: vals=logsumexp(vals,axis=i); fr.pop(i)
        mx=vals.max(); vals=vals-mx; shift+=mx
        if idx+1==len(order) or key(xs[order[idx+1]])!=key(xs[u]): marks.append(shift+float(logsumexp(vals)))
    return (marks[-3]-marks[-3-K])/K/B.W
m=int(sys.argv[1]); W=int(sys.argv[2])
B=Dec(W,24,m); N=4+12*m
for T in [0.05,0.1]:
    for t in [0.9,0.97,1.0,1.03,1.1]:
        x=math.exp((t-1)/T)
        f_full=B.e(t,t,T)
        f_eff=((6*m+3)*t - T*ising_lnZ_cell(B,m,x))/N
        print(f"m={m} W={W} T={T} t={t}: f_full={f_full:.7f}  f_eff(dice Ising)={f_eff:.7f}  diff={f_full-f_eff:+.2e}",flush=True)
