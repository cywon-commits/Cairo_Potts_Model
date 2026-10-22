import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/fam")
from analyze import *
per=math.sqrt(3)/2
for W in [2,3]:
    B=Bulk(build_trunc('dice',W),4*per+0.01,9*per+0.01); h=1e-3
    print(f"W={W}   (p'=expensive cost, q'=mid cost)   prediction: half phase for 1<p'<1+q', full two-colour for p'>1+q'")
    for pp,qq in [(1.1,0.3),(1.2,0.3),(1.4,0.3),(1.6,0.3),(1.3,0.6),(1.5,0.6),(1.7,0.6),(1.9,0.6),(1.5,1.0),(2.2,1.0)]:
        e=B.e(pp,qq); ne=(B.e(pp+h,qq)-B.e(pp-h,qq))/(2*h); nm=(B.e(pp,qq+h)-B.e(pp,qq-h))/(2*h); b=e-pp*ne-qq*nm
        pred="half" if pp<1+qq else "full"
        print(f"   p'={pp:.1f} q'={qq:.1f}: n_exp*12={ne*12:.3f} n_mid*12={nm*12:.3f} b*12={b*12:.3f}   predicted {pred}")
