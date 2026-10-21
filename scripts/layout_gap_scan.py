"""Request 4: scaled-gap scan of the period transfer matrix for column
layouts 'n:S' (see tm_family.parse_layout); one JSON line per point."""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm_family import FamilyTM, parse_layout

P_COST = 0.1


def work(job):
    spec, q, L, T = job
    sel, n, x = parse_layout(spec)
    try:
        t = FamilyTM(L, 1 / T, P_COST, q, sel, n)
        lam = t.spectrum(k=6)
        mu = lam / abs(lam[0])
        xs = L * np.log(1 / np.abs(mu[1:])) / (2 * np.pi * t.P)
        F = -(np.log(abs(lam[0])) - t.beta * t.P * L * t.e0) / (t.P * L)
        return dict(layout=spec, x=x, q=q, L=L, T=T, P=t.P, mu_re=mu.real.tolist(),
                    mu_im=mu.imag.tolist(), gaps=xs.tolist(), F=F)
    except Exception as e:
        return dict(layout=spec, q=q, L=L, T=T, error=repr(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--q", type=float, nargs="+", default=[0.5, 1.05])
    ap.add_argument("--L", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--T", type=float, nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(s, q, L, T) for s in a.layouts for q in a.q for L in a.L for T in a.T]
    jobs.sort(key=lambda j: -j[2])
    with Pool(a.procs) as pool, open(a.out, "a") as fh:
        for r in pool.imap_unordered(work, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
