"""Gap-crossing scan for the order-disorder transition of the interpolating
family (column stripes, x = (n-1)/n) and of Cairo (x = 1) at the same costs.
One JSON line per (model, q, L, T): normalised leading eigenvalues of the
period transfer matrix (k = 0 of the 2-row translation), scaled gaps
x_i = L ln|mu0/mu_i| / (2 pi P), free energy per unit area."""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm_family import FamilyTM, stripe_sel, stripe_sel_low

P_COST = 0.1


def work(job):
    n, q, L, T = job
    # n > 0: x = (n-1)/n stripes; n < 0: x = 1/|n| stripes; n = 0: Cairo (x = 1)
    if n > 0:
        sel, nn = stripe_sel(n)
    elif n < 0:
        sel, nn = stripe_sel_low(-n)
    else:
        sel, nn = (lambda x, y: True), 1
    try:
        t = FamilyTM(L, 1 / T, P_COST, q, sel, nn)
        lam = t.spectrum(k=6)
        mu = lam / abs(lam[0])
        x = L * np.log(1 / np.abs(mu[1:])) / (2 * np.pi * t.P)
        F = -(np.log(abs(lam[0])) - t.beta * t.P * L * t.e0) / (t.P * L)   # per A site, units of kT
        return dict(n=n, q=q, L=L, T=T, P=t.P, mu_re=mu.real.tolist(), mu_im=mu.imag.tolist(),
                    x=x.tolist(), F=F)
    except Exception as e:
        return dict(n=n, q=q, L=L, T=T, error=repr(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[4], help="n>0: x=(n-1)/n; n<0: x=1/|n|; 0: Cairo")
    ap.add_argument("--q", type=float, nargs="+", default=[1.05, 1.10])
    ap.add_argument("--L", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--T", type=float, nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(n, q, L, T) for n in a.n for q in a.q for L in a.L for T in a.T]
    jobs.sort(key=lambda j: -j[2])
    with Pool(a.procs) as pool, open(a.out, "a") as fh:
        for r in pool.imap_unordered(work, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
