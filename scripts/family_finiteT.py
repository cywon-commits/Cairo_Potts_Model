"""Finite-T exact TM for the x = 3/4 family (p = 0.1): n3(q) across the two
T = 0 kinks at q = 1.05 and 1.10, plus leading gaps, on cylinders L."""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm_family import FamilyTM, family_f, stripe_sel

P_COST = 0.1


def work(args):
    L, T, q = args
    sel, n = stripe_sel(4)
    try:
        t = FamilyTM(L, 1 / T, P_COST, q, sel, n)
        f, x = t.free_energy(k=6)
        h = 2e-4
        n3 = (family_f(L, T, P_COST, q + h, sel, n, t.e0) - family_f(L, T, P_COST, q - h, sel, n, t.e0)) / (2 * h)
        return dict(L=L, T=T, q=q, f=f, n3=n3, x=x.tolist())
    except Exception as e:
        return dict(L=L, T=T, q=q, error=repr(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--T", type=float, nargs="+", required=True)
    ap.add_argument("--q", type=float, nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(L, T, q) for L in a.L for T in a.T for q in a.q]
    jobs.sort(key=lambda j: -j[0])
    with Pool(a.procs) as pool, open(a.out, "a") as fh:
        for r in pool.imap_unordered(work, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
