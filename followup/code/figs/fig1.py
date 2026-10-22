import json; from style import *; from matplotlib.patches import Polygon
h=json.load(open(SC+'/fam/hexcheck2_W2.json')); dd=json.load(open(SC+'/ddice/ddice_W2_M120.json'))
panels=[('hub m=4|2','(a) regular: hub lattice, $s=0$',h['hub m=4|2']['pts']),
        ('Cairo|2','(b) kite: Cairo, $s=-1/2$',h['Cairo|2']['pts']),
        ('Cairo + A-B bridge|2','(c) triangle at the bound: Cairo + bridge, $s=-1$',h['Cairo + A-B bridge|2']['pts']),
        ('dd','(d) star: decorated diced, $c_c=3$, $s=-3/2$',dd['pts'])]
fig,ax=plt.subplots(1,4,figsize=(15,4.1))
for a,(k,title,pts) in zip(ax,panels):
    P=[(p,q) for _,p,q in pts]
    a.add_patch(Polygon(P,closed=True,fc=C[0],alpha=0.16,ec=C[0],lw=2))
    a.plot([x for x,y in P],[y for x,y in P],'o',color=C[0],ms=3.5)
    a.add_patch(Polygon(HEX,closed=True,fill=False,ec=INK,lw=1.1,ls=(0,(2,2))))
    a.axhline(0,color=GRID,lw=.8,zorder=0); a.axvline(0,color=GRID,lw=.8,zorder=0)
    a.set_aspect(1); L=3.3; a.set_xlim(-L,L); a.set_ylim(-L,L); a.set_title(title)
    a.set_xlabel('$p=D_2/J$'); 
ax[0].set_ylabel('$q=D_3/J$')
ax[0].plot([],[],'-',color=C[0],label='exact three-colour region (transfer matrix)')
ax[0].plot([],[],ls=(0,(2,2)),color=INK,lw=1.1,label='unit hexagon (Theorem 3 bound)')
fig.legend(loc='lower center',ncol=2,bbox_to_anchor=(0.5,-0.03))
plt.tight_layout(rect=(0,0.05,1,1)); plt.savefig('fig1_gallery.png',bbox_inches='tight'); print('fig1')
