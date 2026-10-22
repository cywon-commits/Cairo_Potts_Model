import sys,math
src=open('ising_map.py').read().split('def ising_lnZ_cell')[0]; exec(src)
def lnZ(B,m,x,h,K=4):
    L=m*np.log(np.array([[2+x,1+x],[1+x,2+x]])); n=B.n; xs=B.xs
    order=sorted(range(n),key=lambda u:(round(xs[u],6),u)); rank={u:k for k,u in enumerate(order)}
    adj=[[] for _ in range(n)]
    for a,b,kind in B.bonds:
        if kind=='u': adj[a].append(b); adj[b].append(a)
    spin=[len(adj[v])>0 for v in range(n)]
    last=[max([rank[w] for w in adj[v]]+[rank[v]]) for v in range(n)]
    fr=[]; vals=np.zeros(()); marks=[]; shift=0.0; key=lambda x_: math.floor((x_+1e-6)/PER)
    for idx,u in enumerate(order):
        if not spin[u]:
            if idx+1==len(order) or key(xs[order[idx+1]])!=key(xs[u]): marks.append(shift+float(logsumexp(vals)))
            continue
        vals=vals[...,None]+np.array([h,-h]); k=vals.ndim
        for v in adj[u]:
            if v in fr:
                i=fr.index(v); sh=[1]*k; sh[i]=2; sh[k-1]=2; vals=vals+L.reshape(sh)
        fr.append(u)
        for i in reversed(range(len(fr))):
            if last[fr[i]]<=idx: vals=logsumexp(vals,axis=i); fr.pop(i)
        mx=vals.max(); vals=vals-mx; shift+=mx
        if idx+1==len(order) or key(xs[order[idx+1]])!=key(xs[u]): marks.append(shift+float(logsumexp(vals)))
    return (marks[-3]-marks[-3-K])/K/B.W
m=int(sys.argv[1]); T=float(sys.argv[2])
Kk=0.25*math.log(3+2*math.sqrt(3)); Kc=-0.5*math.log(math.tanh(Kk)); r=math.exp(2*Kc/m); xc=(2-r)/(r-1); tc=1+T*math.log(xc)
ts=np.round(np.linspace(tc-0.04,tc+0.04,17),4); res={}
for W in [2,3,4,5]:
    B=Dec(W,20,m); h=1e-3; res[W]=[]
    for t in ts:
        x=math.exp((t-1)/T); c=(lnZ(B,m,x,h)+lnZ(B,m,x,-h)-2*lnZ(B,m,x,0))/h**2
        res[W].append(c/W**0.75)   # per-cell chi times W / W^1.75
for a,b in [(2,3),(3,4),(4,5)]:
    d=[res[a][i]-res[b][i] for i in range(len(ts))]
    for i in range(len(ts)-1):
        if d[i]*d[i+1]<0: print(f"eff Ising m={m} T={T}: crossing W={a},{b}: t={ts[i]-d[i]*(ts[i+1]-ts[i])/(d[i+1]-d[i]):.4f} (exact t_c {tc:.4f})",flush=True)
