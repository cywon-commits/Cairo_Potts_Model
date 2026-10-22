import json, math, re, glob, numpy as np; from style import *
Kk=0.25*math.log(3+2*math.sqrt(3)); Kc=-0.5*math.log(math.tanh(Kk))
def tc(m,T): r=math.exp(2*Kc/m); return 1+T*math.log((2-r)/(r-1))
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
a=ax[0]; Ts=np.linspace(0,0.25,60)
pts={}
for f in glob.glob(SC+'/mpar/chi_m*_T*.log'):
    m,T=re.findall(r'chi_m(\d)_T([\d.]+)\.log',f)[0]; L=open(f).read()
    cr=[float(x) for x in re.findall(r'crossing W=3,4: t = ([\d.]+)',L)]
    if cr: pts[(int(m),float(T))]=min(cr,key=lambda x:abs(x-tc(int(m),float(T))))
for i,m in enumerate([2,3,4]):
    a.plot(Ts,[tc(m,T) for T in Ts],color=C[i],label=f'$m={m}$: $t_c=1+T\\ln x_c(m)$, $\\ln x_c={math.log((2-math.exp(2*Kc/m))/(math.exp(2*Kc/m)-1)):+.3f}$')
    P=sorted((T,t) for (mm,T),t in pts.items() if mm==m)
    if P: a.plot(*zip(*P),'o',ms=7,mfc='white',mec=C[i],mew=1.8)
a.plot([],[],'o',ms=7,mfc='white',mec=MUTED,mew=1.8,label='$\\chi/W^{7/4}$ crossing, $W=3,4$ (full decimated model)')
a.axhline(1,color=MUTED,lw=.8,ls=':'); a.text(0.005,0.985,'$m=1$: no transition ($e^{2K_c}>2$)',fontsize=8.5,color=MUTED,va='top')
a.set_xlabel('$T/J$'); a.set_ylabel('$t=D/J$ on the mirror line'); a.set_xlim(0,0.25)
a.set_title('(a) Ising lines leaving the bound corner $(t,T)=(1,0)$'); a.legend(fontsize=7.8,loc='upper left')
b=ax[1]; c=json.load(open(SC+'/mpar/corr.json'))
for i,m in enumerate([2,3]):
    d=np.array([(T,e) for mm,T,e in c if mm==m]); b.semilogy(1/d[:,0],d[:,1],'o',ms=6,color=C[i],label=f'$m={m}$')
    A=np.median(d[:,1]/(d[:,0]*np.exp(-1/d[:,0])))
    x=np.linspace(3,20,50); b.semilogy(x,A*np.exp(-x)/x,color=C[i],lw=1,alpha=.6)
b.plot([],[],color=MUTED,lw=1,label='$A\\,T\\,e^{-J/T}$')
b.set_xlabel('$J/T$'); b.set_ylabel('$|f_{full}-f_{Ising}|$ per site at $t=1$')
b.set_title('(b) the mapping is exact up to $O(T e^{-J/T})$'); b.legend(fontsize=8.5)
plt.tight_layout(); plt.savefig('fig7_ising.png',bbox_inches='tight'); print('fig7',pts)
