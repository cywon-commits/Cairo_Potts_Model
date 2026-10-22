import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math,json; sys.path.insert(0,ROOT+"/ddice")
import numpy as np
from finiteT import FT
which=sys.argv[1]; with_t=(which=="s2"); cell=16 if with_t else 15; d=1.0 if with_t else 1.2
B=FT(2,16,with_t,2*cell); out={}
for T in [0.04,0.08]:
    res=[]
    for u in np.round(np.arange(-5,5.01,0.25),3):
        t=d+u*T; h=1e-3
        n=(B.f(t+h,t+h,T)-B.f(t-h,t-h,T))/(2*h)*cell; res.append((float(u),float(t),float(n)))
        print(which,T,u,n,flush=True)
    out[str(T)]=res
json.dump(out,open(f'fig6_{which}.json','w'))
