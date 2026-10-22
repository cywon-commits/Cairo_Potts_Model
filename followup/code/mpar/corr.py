import sys,math; sys.argv=['x','2','2']
src=open('ising_map.py').read().split('m=int(sys.argv[1])')[0]; exec(src)
import json; out=[]
for m in [2,3]:
    B=Dec(2,24,m); N=4+12*m
    for T in [0.05,0.075,0.1,0.15,0.2,0.25,0.3]:
        t=1.0; x=math.exp((t-1)/T)
        d=abs(B.e(t,t,T)-((6*m+3)*t-T*ising_lnZ_cell(B,m,x))/N)
        out.append((m,T,d)); print(m,T,f"{d:.3e}",f"{d/math.exp(-1/T):.3f}",flush=True)
json.dump(out,open('corr.json','w'))
