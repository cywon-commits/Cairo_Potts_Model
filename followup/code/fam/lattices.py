import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys; sys.path.insert(0,ROOT+"/fam")
from lib import *

def honey_points(R=14, hubs=None):
    pts=[]
    for i in range(-R,R+1):
        for j in range(-R,R+1):
            r=i*a1+j*a2
            pts.append(('A',r)); pts.append(('B',r+np.array([0,A])))
            if hubs and hubs(i,j): pts.append(('H',r+np.array([0.5,A/2])))
    return pts

def honey_rule(t1,t2,d):
    return t1!='H' and t2!='H' and t1!=t2 and abs(np.linalg.norm(d)-A)<1e-6

def hub_rule(angles):
    def rule(t1,t2,d):
        if t1!='H' and t2!='H': return honey_rule(t1,t2,d)
        if t1=='H' and t2=='H': return False
        if abs(np.linalg.norm(d)-A)>1e-6: return False
        dd = d if t1=='H' else -d
        ang=(math.degrees(atan2(dd[1],dd[0]))+360)%360
        return any(abs(ang-x)<1 for x in angles)
    return rule

def dice_rule(t1,t2,d):
    return (t1=='H')!=(t2=='H') and abs(np.linalg.norm(d)-A)<1e-6

def build_trunc(kind,W):
    def f(L):
        RR=int(1.5*L+3*W+8)
        if kind=='honey': pts=honey_points(RR); rule=honey_rule; P=W*a2
        else: pts=honey_points(RR,hubs=lambda i,j:True); rule=dice_rule; P=W*a2
        n,E,xs,typ,pos,disp=cylinder(pts,P,L,rule)
        nn,TE,nx=truncate(n,E,disp,xs)
        return nn,TE,nx
    return f

def build_hub(angles,W,P_dir='A1'):
    A1=a1+a2
    def f(L):
        pts=honey_points(int(1.5*L+4*W+8),hubs=lambda i,j:(i-j)%3==0)
        P=W*A1
        n,E,xs,typ,pos,disp=cylinder(pts,P,L,hub_rule(angles))
        return n,E,xs
    return f
