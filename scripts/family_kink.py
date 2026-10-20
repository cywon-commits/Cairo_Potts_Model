"""T = 0 kinks of E0(q) for the interpolating family x = 3/4 (column stripes
sel = (x mod 4 != 3)), p = 0.10, from the exact min-plus transfer matrix on
cylinders of circumference L.  Left/right derivatives give f_exp = n3 on
each side of every kink."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm_family import ground_energy_family, stripe_sel

p = 0.10
sel, n = stripe_sel(4)
qs = np.round(np.arange(0.02, 1.801, 0.01), 4)
out = {}
for L in [int(a) for a in sys.argv[1:]]:
    E = np.array([ground_energy_family(L, p, q, sel, n, iters=40) for q in qs])
    slope = np.diff(E) / np.diff(qs)            # f_exp on each interval
    kinks = [(float(qs[i + 1]), float(slope[i]), float(slope[i + 1]))
             for i in range(len(slope) - 1) if abs(slope[i + 1] - slope[i]) > 1e-6]
    out[L] = dict(q=qs.tolist(), E0=E.tolist(), kinks=kinks)
    print(f"L={L}: plateaus (q_start, f_exp):")
    start = 0
    for i in range(1, len(slope) + 1):
        if i == len(slope) or abs(slope[i] - slope[start]) > 1e-6:
            print(f"   q in [{qs[start]:.2f}, {qs[i]:.2f}]  f_exp = {slope[start]:.5f}")
            start = i
    sys.stdout.flush()
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data", "family_x34_p0.1_T0_kinks.json"), "w"))
