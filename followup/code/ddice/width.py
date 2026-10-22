import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/ddice")
from finiteT import FT
which=sys.argv[1]; with_t=(which=="s2"); cell=16 if with_t else 15
nP,nM,d=((9,3,1.0) if with_t else (8,3,1.2))
B=FT(2,16,with_t,2*cell)
for T in [0.04,0.08]:
    def n(t,h=1e-3): return (B.f(t+h,t+h,T)-B.f(t-h,t-h,T))/(2*h)*cell
    def cross(frac):
        target=nP-frac*(nP-nM); lo,hi=d-0.6,d+0.6
        for _ in range(16):
            m=(lo+hi)/2
            if n(m)>target: lo=m
            else: hi=m
        return (lo+hi)/2
    t10,t50,t90=cross(0.1),cross(0.5),cross(0.9)
    print(f"{which} T={T}: t10={t10:.4f} t50={t50:.4f} t90={t90:.4f}  width={t90-t10:.4f}  width/T={(t90-t10)/T:.2f}",flush=True)
print("(independent two-level crossover: width/T = 2 ln 9 = 4.39)")
