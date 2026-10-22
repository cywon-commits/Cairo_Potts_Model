# full competitor: arbitrary (improper) 3-colourings; cost b + q n_mid + p n_exp
import sys
from cpsat_check import *
def eall(G,pp,qq):
    m=cp_model.CpModel(); x={(v,c):m.NewBoolVar('') for v in G for c in range(3)}
    for v in G: m.AddExactlyOne(x[v,c] for c in range(3))
    z=[]
    for u,v in G.edges():
        for c in range(3):
            b=m.NewBoolVar(''); m.Add(b>=x[u,c]+x[v,c]-1); z.append(b)
    m.Minimize(SC*sum(z)+sum(int(round(qq*SC))*x[v,1]+int(round(pp*SC))*x[v,2] for v in G)); return solve(m)
rng=np.random.default_rng(300+int(sys.argv[1])); tested=0; viol=0; ex=None; pts=[]
while tested<int(sys.argv[2]):
    n=int(rng.integers(10,19)); G=gen(rng,n,sys.argv[3])
    if not nx.is_connected(G) or e3(G,1,.5) is None: continue
    tested+=1
    for pp,qq in GRID:
        if e3(G,pp,qq)>eall(G,pp,qq)+1e-9: viol+=1; ex=(sorted(G.edges()),pp,qq); pts.append((pp,round(qq,3),(e3(G,pp,qq)-eall(G,pp,qq))/SC))
print("full",sys.argv[1],tested,viol,sorted(set(pts)),flush=True)
