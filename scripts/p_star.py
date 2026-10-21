"""Request 5: sign of the low-T bending of the Potts-M coexistence line.
For each p, q_c(T) from n3(q) = 1/12 (L-cylinder, default L = 8), and
delta_q = q_c(T) - q_0(p), q_0 = (4 - p)/2 the T = 0 boundary.
By Clausius-Clapeyron, dq_c/dT = Delta s / Delta n3 with Delta n3 ~ 1/6 > 0,
so sign(delta_q) = sign(s_Potts - s_M)."""
import json, os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm import n3_density

L = int(os.environ.get("L", 8))
ps = [float(v) for v in sys.argv[2].split(",")]
Ts = [float(v) for v in sys.argv[3].split(",")]
out = []
for p in ps:
    q0 = (4 - p) / 2
    for T in Ts:
        f = lambda q: n3_density(L, T, p, q, h=1e-6) - 1 / 12
        try:
            qc = brentq(f, q0 - 0.08, q0 + 0.12, xtol=1e-9)
        except ValueError:
            qc = np.nan
        row = dict(p=p, T=T, L=L, q0=q0, qc=qc, dq=qc - q0)
        out.append(row)
        print(json.dumps(row), flush=True)
fn = sys.argv[1]
old = json.load(open(fn)) if os.path.exists(fn) else []
json.dump(old + out, open(fn, "w"), indent=1)
