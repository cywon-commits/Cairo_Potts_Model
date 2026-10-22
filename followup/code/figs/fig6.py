import json, math, numpy as np; from style import *
src=open(SC+'/eff/eff.py').read().split('xs_=brentq')[0]; g={'__file__':ROOT+'/eff/eff.py'}; exec(src,g)
dd=json.load(open(SC+'/ddice/fig6_dd.json')); s2=json.load(open(SC+'/ddice/fig6_s2.json'))
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
a=ax[0]
for i,T in enumerate(['0.04','0.08']):
    d=np.array(dd[T]); a.plot(d[:,0],d[:,2],'o-',ms=4,lw=1.5,color=C[i],label=f'$T={T}$')
us=-math.log(2)/5; a.axvline(us,color=MUTED,ls='--',lw=1)
a.text(us+0.15,6.6,'$u^*=-\\Delta s_0/\\Delta n=-\\ln 2/5$\n(residual entropy: $2\\ln2$ vs $3\\ln2$ per cell)',fontsize=8.5,color=MUTED)
a.set_xlabel('$u=(t-d)/T$,  $d=1.2$'); a.set_ylabel('$n_{\\neq\\,\\mathrm{cheap}}$ per cell')
a.set_title('(a) decorated diced, $d=1.2>1$: first-order corner\n(width-2 strip rounds the jump; it sharpens as $T\\to0$ at fixed $u^*$, $t^*=d-T\\ln2/5$)'); a.legend(loc='upper right')
b=ax[1]; u=np.linspace(-5,5,121)
eff=[3+6*g['healed_fraction'](math.exp(x))[0] for x in u]
b.plot(u,eff,color=INK,lw=1.3,label='effective dice Ising $n=3+6h(x)$, $x=e^{u}$')
b.plot(u,9-6/(1+np.exp(-u)),ls=(0,(3,2)),color=MUTED,lw=1.2,label='independent two-level sites')
for i,T in enumerate(['0.04','0.08']):
    d=np.array(s2[T]); b.plot(d[:,0],d[:,2],'o',ms=[8,4][i],mfc=['white',C[i]][i],mec=C[i],mew=1.6,color=C[i],label=f'full model, $T={T}$')
b.axvline(math.log(1.50545),color=C[2],lw=1.2); b.text(0.55,8.3,'midpoint $\\ln x^*=0.409$',fontsize=8.5,color=C[2])
b.set_xlabel('$u=(t-1)/T$,  $d=1$'); b.set_ylabel('$n_{\\neq\\,\\mathrm{cheap}}$ per cell')
b.set_title('(b) $s=-2$ lattice, $d=1$: crossover\n(curves at all $T$ collapse in $u$; width $\\propto T$)'); b.legend(loc='lower left',fontsize=8)
plt.tight_layout(); plt.savefig('fig6_corners.png',bbox_inches='tight'); print('fig6')
