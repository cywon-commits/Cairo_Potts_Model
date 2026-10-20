"""Grid scan of the symmetric-TM spectrum over (q, T) at fixed p for several
circumferences.  Output: one JSON line per (q, T, L) with the leading G
eigenvalues (normalised, signed) and the free energy per unit area."""
import argparse, json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.fss import gaps


def work(args):
    L, T, p, q, k = args
    try:
        g = gaps(L, T, p, q, k=k)
        mu = g["mu"] / np.abs(g["mu"][0])
        return dict(L=L, T=T, p=p, q=q, x=g["x"].tolist(), sign=g["sign"].tolist(),
                    mu_re=mu.real.tolist(), mu_im=mu.imag.tolist(), F=g["F"])
    except Exception as e:  # keep the scan going
        return dict(L=L, T=T, p=p, q=q, error=repr(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, nargs="+", required=True)
    ap.add_argument("--T", type=float, nargs="+", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=[6, 8, 10])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(L, T, a.p, q, a.k) for L in a.L for q in a.q for T in a.T]
    jobs.sort(key=lambda j: -j[0])  # big L first for load balance
    with Pool(a.procs) as pool, open(a.out, "a") as fh:
        for r in pool.imap_unordered(work, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
