import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/fam")
import numpy as np
from lib import *
from lattices import honey_points, honey_rule
def build_hetero(W):
    A1=a1+a2
    def f(L):
        pts=[]
        for i in range(-14,15):
            for j in range(-14,15):
                r=i*a1+j*a2; pts.append(('A',r)); pts.append(('B',r+np.array([0,A])))
                if (i-j)%3==0: pts.append(('H6' if i%2==0 else 'H2', r+np.array([0.5,A/2])))
        def rule(t1,t2,d):
            h1,h2=t1.startswith('H'),t2.startswith('H')
            if not h1 and not h2: return honey_rule(t1,t2,d)
            if h1 and h2: return False
            if abs(np.linalg.norm(d)-A)>1e-6: return False
            ht=t1 if h1 else t2; dd=d if h1 else -d
            ang=(math.degrees(math.atan2(dd[1],dd[0]))+360)%360
            allowed=[30,90,150,210,270,330] if ht=='H6' else [30,210]
            return any(abs(ang-x)<1 for x in allowed)
        n,E,xs,typ,pos,disp=cylinder(pts,W*A1,L,rule)
        return n,E,xs
    return f
B=Bulk(build_hetero(2),3*1.5+0.01,7*1.5+0.01)
h=1e-3
print(" p' (q'=0)   e*7      n_exp*7 (=de/dp)   frustration b*7")
for p in [0.5,0.9,1.1,1.5,2.0,2.5,2.9,3.1,4.0,6.0]:
    e=B.e(p,0.0); ne=(B.e(p+h,0)-B.e(p-h,0))/(2*h); b=e-p*ne
    print(f"  {p:4.1f}     {e*7:7.4f}     {ne*7:7.4f}            {b*7:7.4f}")
print("b_min*7 =",B.e(60.0,0.0)*7)
