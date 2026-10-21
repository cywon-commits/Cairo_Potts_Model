"""PT run for a general column layout 'n:S' with a template order parameter
taken from a T = 0 ground state (template period = ground-state period x 2)."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.mc_family import FamilyPT, OBS
from cairo_potts.tm_family import parse_layout, torus_ground_states

ap = argparse.ArgumentParser()
ap.add_argument("--layout", required=True)
ap.add_argument("--period", type=int, required=True, help="ground-state x-period")
ap.add_argument("--L", type=int, required=True)
ap.add_argument("--q", type=float, required=True)
ap.add_argument("--T", type=float, nargs="+", required=True)
ap.add_argument("--therm", type=int, default=20000)
ap.add_argument("--bins", type=int, default=100)
ap.add_argument("--bin_size", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sel, n, x = parse_layout(a.layout)
if a.L % a.period:
    raise SystemExit("L must be a multiple of the ground-state period")
_, cs = torus_ground_states(a.period, 2, 0.1, a.q, sel, n)
g = cs[0]
tmpl = (g == 0).astype(float) - (g == 1)
run = FamilyPT(a.L, a.T, 0.1, a.q, sel, seed=a.L, template=tmpl)
t0 = time.time()


def save(b, bins):
    np.savez(a.out, bins=bins, Ts=run.Ts, acc=run.acc / np.maximum(run.tries, 1),
             hn3=run.hn3, hE=run.hE, edges_n3=run.H_N3, edges_E=run.H_E, template=tmpl,
             meta=json.dumps(dict(L=a.L, q=a.q, p=0.1, layout=a.layout, x=x, obs=OBS)),
             elapsed=time.time() - t0)
    if b % 20 == 0:
        print(f"bin {b} t={time.time()-t0:.0f}s acc_min {np.min(run.acc/np.maximum(run.tries,1)):.2f}", flush=True)


run.run(a.therm, a.bins, a.bin_size, callback=save)
print("done", time.time() - t0)
