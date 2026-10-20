"""Crossing analysis of the TM scan: for each q and each pair (L, L+2) find
the temperatures where the sector-resolved scaled gaps cross.

  x_o1 : smallest gap in the G-odd sector  (colour-1 density wave, column type)
  x_e1 : smallest gap in the G-even sector (polarisation / crystal-vs-line)
  x_o2 : second gap in the G-odd sector
"""
import json, sys, collections
import numpy as np

fn = sys.argv[1]
rows = [json.loads(l) for l in open(fn)]
rows = [r for r in rows if "error" not in r]
D = collections.defaultdict(dict)
for r in rows:
    x = np.array(r["x"]); s = np.array(r["sign"]); im = np.abs(np.array(r["mu_im"][1:]))
    odd = x[(s < 0) & (im < 1e-9)]; even = x[(s > 0) & (im < 1e-9)]
    D[(r["q"], r["L"])][r["T"]] = dict(
        o1=odd[0] if len(odd) else np.nan, o2=odd[1] if len(odd) > 1 else np.nan,
        e1=even[0] if len(even) else np.nan, F=r["F"])


def crossings(q, L1, L2, key):
    a, b = D[(q, L1)], D[(q, L2)]
    Ts = sorted(set(a) & set(b))
    d = np.array([a[T][key] - b[T][key] for T in Ts])
    out = []
    for i in range(len(Ts) - 1):
        if np.isfinite(d[i]) and np.isfinite(d[i + 1]) and d[i] * d[i + 1] < 0:
            T0 = Ts[i] - d[i] * (Ts[i + 1] - Ts[i]) / (d[i + 1] - d[i])
            w = (T0 - Ts[i]) / (Ts[i + 1] - Ts[i])
            xv = (1 - w) * b[Ts[i]][key] + w * b[Ts[i + 1]][key]
            out.append((round(T0, 4), round(xv, 3), "+" if d[i] > 0 else "-"))
    return out


qs = sorted({k[0] for k in D})
Ls = sorted({k[1] for k in D})
summary = {}
for q in qs:
    line = []
    for key in ("o1", "o2", "e1"):
        for L1, L2 in zip(Ls[:-1], Ls[1:]):
            c = crossings(q, L1, L2, key)
            summary[(q, key, L1)] = c
            line.append(f"{key}({L1},{L2})={c}")
    print(f"q={q}:\n   " + "\n   ".join(line))
