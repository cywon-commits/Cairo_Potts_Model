import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/ddice")
from finiteT import FT
T=float(sys.argv[1]); M=int(sys.argv[2])
B=FT(2,M,True,32)
print(f"T={T} M={M}:   t    n*16 (TM)   rhoM logistic   tau/M step")
for t in [0.85,0.9,0.95,0.98,1.0,1.02,1.04,1.06,1.1,1.15]:
    h=1e-3; n=(B.f(t+h,t+h,T)-B.f(t-h,t-h,T))/(2*h)*16
    lg=9-6/(1+math.exp(-(t-1)/T))
    print(f"        {t:.2f}   {n:7.3f}      {lg:7.3f}        {'9' if t<1 else '3'}",flush=True)
