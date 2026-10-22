from style import *; import numpy as np
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
# (a) composition plane for Cairo: three-colour phase Z and two-colour phase G
a=ax[0]
Z=(5/12,1/6); G=(1/3,0.0)
a.plot(*Z,'o',ms=11,color=C[0]); a.annotate('Z: proper three-colouring\n$B=0$, $(f_{mid},f_{exp})=(5/12,1/6)$',Z,xytext=(0.43,0.21),fontsize=9)
a.plot(*G,'s',ms=10,color=C[1]); a.annotate('adjacent phase: two colours\n$B=b_{min}=1/3$, $(g,0)=(1/3,0)$',G,xytext=(0.18,0.035),fontsize=9)
a.annotate('',xy=G,xytext=Z,arrowprops=dict(arrowstyle='->',color=MUTED,lw=1.5))
a.text(0.395,0.075,'move $\\Delta k=(-1/12,-1/6)$\ncosts $\\Delta B=1/3$ → edge C2',fontsize=8.5,color=MUTED)
a.text(0.43,0.115,'local move (not to scale): unheal one\nexpensive site, $\\Delta N_{exp}=-1$,\n$\\Delta(NB)=c_c=2$ → edge C1',fontsize=8.5,color=C[2])
a.annotate('',xy=(0.425,1/6-0.03),xytext=Z,arrowprops=dict(arrowstyle='->',color=C[2],lw=1.5))
a.set_xlim(0.15,0.62); a.set_ylim(-0.02,0.26); a.set_xlabel('$n_{mid}$ per site'); a.set_ylabel('$n_{exp}$ per site')
a.set_title('(a) composition plane: phases are points of the convex function $B(k)$')
# (b) fundamental sector: each move = one half-plane
b=ax[1]; q=np.linspace(0,2.4,50)
poly=[(0,0),(2,0),(4/3,4/3),(0,0)]
from matplotlib.patches import Polygon
b.add_patch(Polygon([(0,0),(2,0),(4/3,4/3)],closed=True,fc=C[0],alpha=0.16,ec='none'))
qq=np.linspace(0,1.6,10); b.plot(2-qq/2,qq,color=C[1],label="C2 (adjacent phase): $p'=c_c+s\\,q'$, $s=\\frac{n^c_{mid}-f_{mid}}{f_{exp}-n^c_{exp}}=-1/2$")
b.axvline(2,color=C[2],ls='--',lw=1.6,label="C1 (local cluster): $p'=c_c=2$")
b.plot([0,2.2],[0,2.2],':',color=INK,lw=1,label='mirror line $q\'=p\'$')
b.plot(4/3,4/3,'o',ms=9,mfc='none',mec=C[1],mew=2); b.text(0.12,1.6,'mirror corner (circle): $d=c_c/(1-s)=4/3$',fontsize=8.5)
b.add_patch(Polygon([(0,0),(1,0),(1,1)],closed=True,fill=False,ec=INK,lw=1.1,ls=(0,(2,2))))
b.text(0.62,0.12,'unit-hexagon\nsector',fontsize=8,color=MUTED)
b.set_xlim(0,2.6); b.set_ylim(0,1.8); b.set_aspect(1); b.set_xlabel("$p'$ (largest cost)"); b.set_ylabel("$q'$ (middle cost)")
b.set_title('(b) cost plane: each phase of (a) bounds one edge')
b.legend(loc='upper center',bbox_to_anchor=(0.5,-0.16),fontsize=8.5)
plt.tight_layout(); plt.savefig('fig2_duality.png',bbox_inches='tight'); print('fig2')
