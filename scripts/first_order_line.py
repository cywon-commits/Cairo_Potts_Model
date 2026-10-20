"""Locate the Potts / M coexistence line q_c(T) at p = 1 from the jump of
n3 = df/dq on cylinders (bisection on n3 = 1/12, halfway between the
Potts value 1/6 and the M value 0 at low T), and record the jump size
from n3 just below / above."""
import json, os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm import n3_density

out = []
for T in [float(t) for t in sys.argv[2:]]:
    for L in (6, 8, 10):
        f = lambda q: n3_density(L, T, 1.0, q, h=2e-5) - 1 / 12
        try:
            qc = brentq(f, 1.40, 1.505, xtol=2e-5)
        except ValueError:
            qc = np.nan
        row = dict(T=T, L=L, qc=qc)
        if np.isfinite(qc):
            row["n3_lo"] = n3_density(L, T, 1.0, qc - 0.004)
            row["n3_hi"] = n3_density(L, T, 1.0, qc + 0.004)
        out.append(row)
        print(json.dumps(row), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
