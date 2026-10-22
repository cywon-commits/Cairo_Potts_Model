import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys,math; sys.path.insert(0,ROOT+"/fam")
from analyze import *
per=math.sqrt(3)/2
import json
B=Bulk(build_trunc('dice',2),4*per+0.01,9*per+0.01); h=1e-3; out=[]
for qq in np.round(np.arange(0,1.21,0.1),3):
  for pp in np.round(np.arange(max(qq,0.5),2.81,0.05),3):
    ne=(B.e(pp+h,qq)-B.e(pp-h,qq))/(2*h); out.append((pp,qq,ne*12))
json.dump(out,open('half_grid.json','w')); print('done')
