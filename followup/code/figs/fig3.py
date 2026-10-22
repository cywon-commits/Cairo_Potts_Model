import json, itertools, numpy as np, networkx as nx; from style import *
from matplotlib.colors import ListedColormap
fig,ax=plt.subplots(1,3,figsize=(15,4.6),gridspec_kw={'width_ratios':[1,1,1.05]})
# (a) heterogeneous hub
d=np.array(json.load(open(SC+'/proof/hetero_curve.json')))
a=ax[0]; a.plot(d[:,0],d[:,2],color=C[0],label='$7\\,n_{exp}$ (expensive sites per cell)')
a.plot(d[:,0],d[:,3],color=C[1],label='$7\\,b$ (frustrated bonds per cell)')
for x,lab in [(1,'2-fold hub\nunheals ($z_T/2=1$)'),(3,'6-fold hub\nunheals ($z_T/2=3$)')]:
    a.axvline(x,color=MUTED,lw=1,ls='--'); a.text(x+0.05,1.15 if x==1 else 0.8,lab,fontsize=8.5,color=MUTED)
a.set_xlabel("$p'$ (with $q'=0$)"); a.set_ylabel('count per cell'); a.set_ylim(-0.05,2.15)
a.set_title('(a) heterogeneous hubs: intermediate phase on the axis\n(Theorem 1: threshold = marginal cost of each hub)')
a.legend(loc='upper left',fontsize=8.5)
# (b) truncated diced half phase
g=json.load(open(SC+'/proof/half_grid.json')); b=ax[1]
for p,q,n in g:
    cls=0 if n>1.75 else (1 if n>0.75 else 2)
    if abs(n-round(n))>0.1 and n<1.9: continue
    b.plot(p,q,'s',ms=4.2,color=[C[0],C[3],C[1]][cls],mew=0)
qq=np.linspace(0,1.25,10); b.plot(1+qq,qq,color=INK,lw=1.2); b.axvline(1,color=INK,lw=1.2,ymax=1.25/1.3)
b.plot([0,1.3],[0,1.3],':',color=INK,lw=1)
for i,lab in enumerate(['three-colour ($12n_{exp}=2$)','half phase ($12n_{exp}=1$)','two-colour ($n_{exp}=0$)']): b.plot([],[],'s',ms=7,color=[C[0],C[3],C[1]][i],label=lab)
b.text(0.92,1.24,"$p'=1$",fontsize=8.5,ha='right'); b.text(2.27,1.22,"$p'=1+q'$",fontsize=8.5)
b.set_xlim(0.45,2.85); b.set_ylim(0,1.3); b.set_xlabel("$p'$"); b.set_ylabel("$q'$")
b.set_title('(b) truncated diced: half-phase wedge $1<p\'<1+q\'$')
b.legend(loc='upper left',fontsize=8,bbox_to_anchor=(0,-0.15),ncol=2)
# (c) 9-vertex counterexample
E=[(0,1),(0,2),(0,3),(0,4),(0,6),(0,8),(1,2),(1,4),(2,6),(3,8),(4,5),(4,7),(5,6),(6,7),(7,8)]
G=nx.Graph(E); w=(1,0,0,0,2,0,2,0,2)
best=None
for col in itertools.product(range(3),repeat=9):
    if any(col[u]==col[v] for u,v in E): continue
    c=sum(1 for x in col if x); 
    if best is None or c<best[0]: best=(c,col)
pos=nx.kamada_kawai_layout(G)
c=ax[2]; cols=['#ffffff',C[0],C[1]]
for u,v in E:
    fr=w[u]==w[v]; c.plot(*zip(pos[u],pos[v]),color=C[1] if fr else MUTED,lw=3 if fr else 1.3,zorder=1)
for v in G: 
    c.scatter(*pos[v],s=330,color=cols[w[v]],edgecolor=INK,lw=1.2,zorder=2); c.text(*pos[v],str(v),ha='center',va='center',fontsize=8,color=INK if w[v]==0 else 'white',zorder=3)
c.set_aspect(1); c.axis('off')
c.set_title(f'(c) mixed competitor beats every proper colouring\non the mirror ray at $t=1$: $b+n_{{\\neq 0}}=1+4=5<{best[0]}$')
c.scatter([],[],s=80,color='white',edgecolor=INK,label='colour 0 (free)'); c.scatter([],[],s=80,color=C[0],edgecolor=INK,label='colour 1 (cost $t$)')
c.scatter([],[],s=80,color=C[1],edgecolor=INK,label='colour 2 (cost $t$)'); c.plot([],[],color=C[1],lw=3,label='frustrated bond')
c.legend(loc='upper left',bbox_to_anchor=(0,-0.02),ncol=2,fontsize=8)
plt.tight_layout(); plt.savefig('fig3_theorems.png',bbox_inches='tight'); print('fig3',best)
