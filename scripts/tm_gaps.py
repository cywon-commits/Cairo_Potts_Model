"""Scaled gaps x_i(L) = L ln(mu0/|mu_i|) / 2pi of the symmetric column
transfer G on cylinders L = 4..Lmax, over a temperature grid."""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm import SymmetricTM

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=float, default=1.0)
ap.add_argument("--q", type=float, required=True)
ap.add_argument("--L", type=int, nargs="+", default=[4, 6, 8, 10])
ap.add_argument("--T", type=float, nargs="+", required=True)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--out", required=True)
a = ap.parse_args()

res = []
for L in a.L:
    for T in a.T:
        ev = SymmetricTM(L, 1 / T, a.p, a.q).spectrum_G(k=a.k)
        mu = ev / np.abs(ev[0])
        x = L * np.log(1 / np.abs(mu[1:])) / (2 * np.pi)
        res.append(dict(L=L, T=T, mu_re=mu.real.tolist(), mu_im=mu.imag.tolist(), x=x.tolist()))
        print(L, T, np.round(mu.real, 4), np.round(x[:4], 4), flush=True)
json.dump(dict(p=a.p, q=a.q, rows=res), open(a.out, "w"))
