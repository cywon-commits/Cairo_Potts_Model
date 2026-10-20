"""Exact ground-state entropy (A configurations x optimal B multiplicities)
of the x = 3/4 family, p = 0.1, on Lx x Ly tori; per-site entropy from the
slope in Lx.  Excess entropy at a kink = local, interface-free switching."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm_family import torus_ground_count, stripe_sel

sel, n = stripe_sel(4)
Ly = int(sys.argv[1]); Lxs = [int(a) for a in sys.argv[2].split(",")]
out = []
for q in (1.02, 1.05, 1.075, 1.10, 1.15):
    lg = [torus_ground_count(Lx, Ly, 0.1, q, sel, n)[1] for Lx in Lxs]
    s = (lg[-1] - lg[-2]) / (3 * Ly * (Lxs[-1] - Lxs[-2]))
    out.append(dict(q=q, Ly=Ly, Lx=Lxs, lnGS=lg, s0=s))
    print(f"Ly={Ly} q={q:.3f} lnGS={np.round(lg, 3).tolist()} s0={s:.4f}", flush=True)
fn = os.path.join(os.path.dirname(__file__), "..", "data", "family_gs_entropy.json")
old = json.load(open(fn)) if os.path.exists(fn) else []
json.dump(old + out, open(fn, "w"), indent=1)
