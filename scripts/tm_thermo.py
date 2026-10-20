"""Exact (cylinder) thermodynamics from the transfer matrix: free energy f,
entropy s = -df/dT, specific heat c = -T d^2f/dT^2, densities n2, n3
(all per site, N = 3 N_A), on a T grid, for given (p, q, L)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm import free_energy

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=float, default=1.0)
ap.add_argument("--q", type=float, nargs="+", required=True)
ap.add_argument("--L", type=int, default=8)
ap.add_argument("--T", type=float, nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
res = []
for q in a.q:
    for T in a.T:
        h = 1e-3 * T
        f0, fp, fm = (free_energy(a.L, t, a.p, q) for t in (T, T + h, T - h))
        hq = 1e-5
        n3 = (free_energy(a.L, T, a.p, q + hq) - free_energy(a.L, T, a.p, q - hq)) / (2 * hq)
        n2 = (free_energy(a.L, T, a.p + hq, q) - free_energy(a.L, T, a.p - hq, q)) / (2 * hq)
        s = -(fp - fm) / (2 * h)
        c = -T * (fp - 2 * f0 + fm) / h ** 2
        r = dict(p=a.p, q=q, L=a.L, T=T, f=f0, s=s, c=c, e=f0 + T * s, n3=n3, n2=n2)
        res.append(r)
        print(json.dumps(r), flush=True)
json.dump(res, open(a.out, "w"))
