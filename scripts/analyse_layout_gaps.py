"""Crossings x_0(L) = x_0(L') (leading gap) per layout and q; also reports
whether the gap grows with L at every T (disordered) or shrinks (ordered)."""
import json, os, sys, collections
import numpy as np
fn = sys.argv[1]
D = collections.defaultdict(dict)
for l in open(fn):
    r = json.loads(l)
    if "error" in r:
        continue
    D[(r["layout"], r["q"], r["L"])][r["T"]] = r["gaps"][0]
out = []
for (lay, q) in sorted({k[:2] for k in D}):
    Ls = sorted(k[2] for k in D if k[:2] == (lay, q))
    line = f"{lay:10s} q={q}: "
    for L1, L2 in zip(Ls[:-1], Ls[1:]):
        a, b = D[(lay, q, L1)], D[(lay, q, L2)]
        Ts = sorted(set(a) & set(b))
        d = np.array([a[T] - b[T] for T in Ts])
        cr = [Ts[i] - d[i] * (Ts[i + 1] - Ts[i]) / (d[i + 1] - d[i])
              for i in range(len(Ts) - 1) if d[i] > 0 and d[i + 1] < 0 and b[Ts[i]] > 0.01]
        state = "ordered at low T" if d[0] > 0 else "gap grows with L at all T" if (d < 0).all() else ""
        line += f"({L1},{L2}) T*={[round(c, 4) for c in cr]} {state};  "
        out.append(dict(layout=lay, q=q, L1=L1, L2=L2, Tstar=cr, all_disordered=bool((d < 0).all())))
    print(line)
json.dump(out, open(fn.replace(".jsonl", "_crossings.json"), "w"), indent=1)
