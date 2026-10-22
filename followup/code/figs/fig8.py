import json, math, numpy as np; from style import *
from matplotlib.patches import Polygon
P2=SC+'/proof2'
fig,ax=plt.subplots(1,3,figsize=(15,4.9),gridspec_kw={'width_ratios':[0.8,1.1,1]})
# (a) motif + hub move
a=ax[0]; a.set_aspect(1); a.axis('off')
hub={'A':(0,0),'B':(1,0),'C':(0.5,math.sqrt(3)/2)}
col={'A':C[1],'B':'#ffffff','C':C[0]}
lab={'A':'3','B':'1','C':'2'}
m=3
for u,v in [('A','B'),('B','C'),('C','A')]:
    P,Q=np.array(hub[u],float),np.array(hub[v],float); a.plot(*zip(P,Q),color=MUTED,lw=1.6)
    mid=(P+Q)/2; nrm=np.array([-(Q-P)[1],(Q-P)[0]]); nrm/=np.linalg.norm(nrm)
    cen=np.array([0.5,math.sqrt(3)/6]); 
    if np.dot(mid-cen,nrm)<0: nrm=-nrm
    third=({'A','B','C'}-{u,v}).pop()
    for k in range(m):
        s=mid+nrm*(0.16+0.13*k)
        a.plot(*zip(P,s),color=GRID,lw=1); a.plot(*zip(Q,s),color=GRID,lw=1)
        a.scatter(*s,s=60,color=col[third],edgecolor=INK,lw=.8,zorder=3)
for h,(x,y) in hub.items():
    a.scatter(x,y,s=260,color=col[h],edgecolor=INK,lw=1.2,zorder=4); a.text(x,y,lab[h],ha='center',va='center',fontsize=9,color='white' if h!='B' else INK,zorder=5)
a.text(0.5,-0.55,"triangular hubs + m sites on each hub bond,\neach joined to both ends (m = 3 shown).\nHub move: recolour one '2' hub to '1'\n→ 3 frustrated bonds, 3m expensive sites released\n→ axis threshold $p'_c=1/m$",ha='center',va='top',fontsize=8.6)
a.set_xlim(-0.6,1.6); a.set_ylim(-1.35,1.45); a.set_title('(a) decorated triangular lattice')
# (b) regions
b=ax[1]
for i,mm in enumerate([1,2,3]):
    d=json.load(open(f'{P2}/tri_region_m{mm}.json')); Pt=[(p,q) for _,p,q in d]
    b.add_patch(Polygon(Pt,closed=True,fc=C[i],alpha=0.12,ec='none')); b.plot([x for x,y in Pt+Pt[:1]],[y for x,y in Pt+Pt[:1]],color=C[i],lw=2)
    b.plot([],[],color=C[i],lw=2,label=f'$m={mm}$: axis $1/m$, mirror $1/(2m-1/3)$')
b.add_patch(Polygon(HEX,closed=True,fill=False,ec=INK,lw=1.1,ls=(0,(2,2))))
b.plot([],[],ls=(0,(2,2)),color=INK,lw=1.1,label='unit hexagon (former Theorem 3)')
b.axhline(0,color=GRID,lw=.8,zorder=0); b.axvline(0,color=GRID,lw=.8,zorder=0)
b.set_aspect(1); b.set_xlim(-1.2,1.2); b.set_ylim(-1.2,1.2); b.set_xlabel('$p=D_2/J$'); b.set_ylabel('$q=D_3/J$')
b.set_title('(b) exact three-colour regions fall inside the hexagon'); b.legend(fontsize=7.8,loc='upper left',bbox_to_anchor=(0,-0.13))
# (c) Y_s
c=ax[2]; s=np.array([1,2,3,4,6]); c.plot(s,3/(4*s),'o-',color=C[0],label='$Y_s$ (14 → 74 sites, max degree 5): $3/(4s)$')
L=np.array([6,12,18]); c.plot(L/3,9/(4*L),'s--',color=C[1],label='periodic $Y$ strip, period $L$: $9/(4L)$ (vs $L/3$)')
c.axhline(1,color=INK,lw=1,ls=(0,(2,2))); c.text(1.0,1.03,'unit-hexagon bound',fontsize=8.5)
c.set_xlabel('$s$ (strip length / 3)'); c.set_ylabel("axis threshold $p'_c$ = (min bipartization)/(min independent OCT)")
c.set_ylim(0,1.15); c.set_title('(c) frozen colourings: threshold → 0 at bounded degree'); c.legend(fontsize=7.8,loc='center right')
plt.tight_layout(); plt.savefig('fig8_counterexamples.png',bbox_inches='tight'); print('ok')
