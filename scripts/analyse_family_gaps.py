"""Crossings of the scaled gaps x_i(L) between successive L for the family
gap scan, and central charge from F(L) = F_inf - pi c / (6 L^2)."""
import json, os, sys, collections
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "family_gap_scan.jsonl")
D = collections.defaultdict(dict)
for l in open(fn):
    r = json.loads(l)
    if "error" in r:
        continue
    D[(r["n"], r["q"], r["L"])][r["T"]] = r
out = []
for (n, q) in sorted({(k[0], k[1]) for k in D}):
    Ls = sorted(k[2] for k in D if k[:2] == (n, q))
    for lev in (0, 1, 2):
        for L1, L2 in zip(Ls[:-1], Ls[1:]):
            a, b = D[(n, q, L1)], D[(n, q, L2)]
            Ts = sorted(set(a) & set(b))
            xa = np.array([a[T]["x"][lev] for T in Ts]); xb = np.array([b[T]["x"][lev] for T in Ts])
            d = xa - xb
            for i in range(len(Ts) - 1):
                if d[i] > 0 and d[i + 1] < 0 and xb[i] > 0.01:
                    T0 = Ts[i] - d[i] * (Ts[i + 1] - Ts[i]) / (d[i + 1] - d[i])
                    x0 = np.interp(T0, Ts, xb)
                    sgn = np.sign(np.interp(T0, Ts, [b[T]["mu_re"][lev + 1] for T in Ts]))
                    out.append(dict(n=n, q=q, level=lev, L1=L1, L2=L2, T=T0, x=x0, sign=sgn))
                    print(f"n={n} q={q} level {lev} ({'+' if sgn > 0 else '-'}) ({L1},{L2}): T*={T0:.4f} x*={x0:.3f}")
json.dump(out, open(os.path.join(ROOT, "data", "family_gap_crossings.json"), "w"), indent=1)
