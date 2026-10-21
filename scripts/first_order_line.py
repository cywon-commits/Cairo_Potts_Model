"""Locate the Potts / M coexistence line q_c(T) at fixed p (default 1) from the jump of
n3 = df/dq on cylinders (bisection on n3 = 1/12, halfway between the
Potts value 1/6 and the M value 0 at low T), and record the jump size
from n3 just below / above."""
import json, os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm import n3_density

# usage: first_order_line.py out.json [--p P --qlo A --qhi B] T1 T2 ...
args = sys.argv[2:]
P, QLO, QHI = 1.0, 1.40, 1.505
while args and args[0].startswith("--"):
    k, v = args[0], float(args[1]); args = args[2:]
    if k == "--p": P = v
    elif k == "--qlo": QLO = v
    elif k == "--qhi": QHI = v
out = []
for T in [float(t) for t in args]:
    for L in (6, 8, 10):
        f = lambda q: n3_density(L, T, P, q, h=2e-5) - 1 / 12
        try:
            qc = brentq(f, QLO, QHI, xtol=2e-5)
        except ValueError:
            qc = np.nan
        row = dict(p=P, T=T, L=L, qc=qc)
        if np.isfinite(qc):
            row["n3_lo"] = n3_density(L, T, P, qc - 0.004)
            row["n3_hi"] = n3_density(L, T, P, qc + 0.004)
        out.append(row)
        print(json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
