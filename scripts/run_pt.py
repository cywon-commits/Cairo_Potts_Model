"""Parallel-tempering production run for one (p, q, L).

1. pilot on a coarse ladder -> energy variance sigma_E(T) of the total
   A-marginal log-weight proxy (total energy fluctuations),
2. ladder with constant  d(beta) * sigma_E  (target swap rate ~ 30 %),
3. production; bins of observables saved to an .npz after every bin.
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.mc import PTRun, OBS

ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, required=True)
ap.add_argument("--p", type=float, default=1.0)
ap.add_argument("--q", type=float, required=True)
ap.add_argument("--Tmin", type=float, required=True)
ap.add_argument("--Tmax", type=float, required=True)
ap.add_argument("--pilot_R", type=int, default=12)
ap.add_argument("--pilot_steps", type=int, default=3000)
ap.add_argument("--dbs", type=float, default=1.1, help="d(beta)*sigma_E per gap")
ap.add_argument("--therm", type=int, default=20000)
ap.add_argument("--bins", type=int, default=100)
ap.add_argument("--bin_size", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ladder", type=float, nargs="*", help="explicit temperatures (skip pilot)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
N = 3 * a.L ** 2

if a.ladder:
    Ts = np.array(sorted(a.ladder))
else:
    Tp = np.geomspace(a.Tmin, a.Tmax, a.pilot_R)
    pilot = PTRun(a.L, Tp, a.p, a.q, seed=a.seed + 1000)
    b = pilot.run(a.pilot_steps // 2, 4, a.pilot_steps // 8)
    E, E2, Ev = (b[:, :, OBS.index(k)].mean(0) for k in ("E", "E2", "Evar"))
    sig = np.sqrt(np.maximum(E2 - E ** 2 + Ev, 1e-12)) * N  # total-energy std
    # integrate d(beta)/ds = dbs / sigma_E(beta) from beta_max down to beta_min
    bp = 1 / Tp[::-1]; sp = sig[::-1]
    betas = [bp[0]]
    while betas[-1] < bp[-1]:
        s = np.interp(betas[-1], bp, sp)
        betas.append(betas[-1] + a.dbs / s)
    betas[-1] = bp[-1]
    Ts = np.sort(1 / np.array(betas))
    print("pilot sigma_E", np.round(sig, 1), "-> R =", len(Ts), flush=True)

run = PTRun(a.L, Ts, a.p, a.q, seed=a.seed)
t0 = time.time()
meta = dict(L=a.L, p=a.p, q=a.q, Ts=Ts.tolist(), obs=OBS, bin_size=a.bin_size, therm=a.therm)


def save(bidx, bins):
    np.savez(a.out, bins=bins, Ts=Ts, acc=run.acc / np.maximum(run.tries, 1),
             trips=run.trips, meta=json.dumps(meta), elapsed=time.time() - t0)
    if bidx % 10 == 0:
        print(f"bin {bidx} t={time.time()-t0:.0f}s acc min {np.min(run.acc/np.maximum(run.tries,1)):.2f}"
              f" trips {run.trips.sum():.0f}", flush=True)


run.run(a.therm, a.bins, a.bin_size, callback=save)
print("done", time.time() - t0)
