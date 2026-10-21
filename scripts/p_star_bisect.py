"""p*(T): zero of delta_q(p, T) = q_c(T; p) - (4 - p)/2 (L = 8 cylinder)."""
import json, os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm import n3_density

L = 8


def dq(p, T):
    q0 = (4 - p) / 2
    return brentq(lambda q: n3_density(L, T, p, q, h=1e-6) - 1 / 12, q0 - 0.08, q0 + 0.12, xtol=1e-10) - q0


out = []
for T in [float(v) for v in sys.argv[2].split(",")]:
    ps = brentq(lambda p: dq(p, T), 0.85, 0.97, xtol=2e-4)
    out.append(dict(T=T, p_star=ps))
    print(json.dumps(out[-1]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
