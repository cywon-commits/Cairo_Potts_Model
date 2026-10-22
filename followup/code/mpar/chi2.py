import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/mpar")
import numpy as np
from decim import Dec
Kk=0.25*math.log(3+2*math.sqrt(3)); Kc=-0.5*math.log(math.tanh(Kk))
m=int(sys.argv[1]); T=float(sys.argv[2])
r=math.exp(2*Kc/m); xc=(2-r)/(r-1); tc=1+T*math.log(xc)
print(f"m={m} T={T}: predicted t_c = {tc:.5f}")
ts=np.round(np.linspace(tc-0.04,tc+0.04,9),4)
res={}
for W in [2,3,4]:
    B=Dec(W,20,m); N=4+12*m; d=2e-5
    res[W]=[]
    for t in ts:
        f0=B.e(t,t,T); fp=B.e(t+d,t-d,T); fm=B.e(t-d,t+d,T)
        chi=-(fp+fm-2*f0)/d**2*N/T
        res[W].append(chi)
    print(f"W={W}: chi/W^1.75 = "+"  ".join(f"{t:.4f}:{c/W**1.75:8.2f}" for t,c in zip(ts,res[W])),flush=True)
for a,b in [(2,3),(3,4)]:
    diff=[res[a][i]/a**1.75-res[b][i]/b**1.75 for i in range(len(ts))]
    for i in range(len(ts)-1):
        if diff[i]*diff[i+1]<0:
            tx=ts[i]-diff[i]*(ts[i+1]-ts[i])/(diff[i+1]-diff[i]); print(f"crossing W={a},{b}: t = {tx:.4f}   (predicted {tc:.4f})")
