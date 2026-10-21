"""PT run for the x = 3/4 family (p = 0.1) at fixed q; saves binned
observables and n3 / E histograms per temperature."""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.mc_family import FamilyPT, OBS
from cairo_potts.tm_family import stripe_sel

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, required=True)
ap.add_argument("--q", type=float, required=True)
ap.add_argument("--T", type=float, nargs="+", required=True)
ap.add_argument("--therm", type=int, default=20000)
ap.add_argument("--bins", type=int, default=100)
ap.add_argument("--bin_size", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.L % 8:
    print("warning: L not a multiple of 8 frustrates the x = 3/4 ground state")
sel, n = stripe_sel(4)
tmpl = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "family_template_8x2.npy"))
run = FamilyPT(a.L, a.T, 0.1, a.q, sel, seed=a.L, template=tmpl)
t0 = time.time()


def save(b, bins):
    np.savez(a.out, bins=bins, Ts=run.Ts, acc=run.acc / np.maximum(run.tries, 1),
             hn3=run.hn3, hE=run.hE, edges_n3=run.H_N3, edges_E=run.H_E,
             meta=json.dumps(dict(L=a.L, q=a.q, p=0.1, x="3/4 stripes", obs=OBS)),
             elapsed=time.time() - t0)
    if b % 20 == 0:
        print(f"bin {b} t={time.time()-t0:.0f}s acc {np.round(run.acc/np.maximum(run.tries,1),2).tolist()}", flush=True)


run.run(a.therm, a.bins, a.bin_size, callback=save)
print("done", time.time() - t0)
