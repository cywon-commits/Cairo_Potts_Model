"""Longer explicit-B MC runs (4 independent seeds, 2e6 sweeps each) on the
6 x 6 torus for verification V3b; exact values from data/verification.json."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.explicit import explicit_mc

here = os.path.join(os.path.dirname(__file__), "..", "data")
ref = {r["T"]: r for r in json.load(open(os.path.join(here, "verification.json")))["V3bc_6x6_torus"]}
res = []
for T in (0.2, 0.15):
    Es, ns = [], []
    for seed in range(4):
        E, n3 = explicit_mc(6, 6, T, 1, 1.2, n_sweeps=2_000_000, n_therm=20000, seed=100 + seed, n_bins=40)
        Es.append(E); ns.append(n3)
    Es = np.array(Es); ns = np.array(ns)
    seed_means_E = Es.mean(1); seed_means_n = ns.mean(1)
    row = dict(T=T, E_exact=ref[T]["E_exact"], n3_exact=ref[T]["n3_exact"],
               E=seed_means_E.mean(), E_err=seed_means_E.std(ddof=1) / 2,
               n3=seed_means_n.mean(), n3_err=seed_means_n.std(ddof=1) / 2,
               E_seeds=seed_means_E.tolist(), n3_seeds=seed_means_n.tolist())
    print(json.dumps(row), flush=True)
    res.append(row)
json.dump(res, open(os.path.join(here, "verification_explicit_long.json"), "w"), indent=1)
