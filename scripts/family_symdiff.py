"""Cheap first step of request 2: optimal configurations on both sides of
each T = 0 kink (x = 3/4, p = 0.1) on Lx x 4 tori (Lx multiple of 8) and the
minimal symmetric difference of the colour-3 sets."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm_family import (torus_ground_states, colour3_sets, stripe_sel,
                                   local_energy_tables, _kind_index)

p = 0.1
sel, n = stripe_sel(4)


def energy(a, q):
    em = local_energy_tables(p, q); Lx, L = a.shape
    return sum(em[_kind_index(x, y, sel)][a[x, y], a[(x + 1) % Lx, y], a[x, (y + 1) % L],
               a[(x + 1) % Lx, (y + 1) % L]] for x in range(Lx) for y in range(L)) / (3 * Lx * L)


out = []
for Lx in (8, 16, 24):
    N = 3 * Lx * 4
    res = {}
    for q in (1.02, 1.075, 1.15):
        e, cs = torus_ground_states(Lx, 4, p, q, sel, n)
        assert all(abs(energy(a, q) - e) < 1e-12 for a in cs), "reconstruction"
        sets = [colour3_sets(a, p, q, sel) for a in cs]
        res[q] = (sets, cs)
        print(f"Lx={Lx} q={q:.3f} E/N={e:.6f} #found={len(cs)} n3*N={sorted({len(s) for s in sets})}", flush=True)
    for q1, q2 in ((1.02, 1.075), (1.075, 1.15)):
        d = min(len(s1 ^ s2) for s1 in res[q1][0] for s2 in res[q2][0])
        out.append(dict(Lx=Lx, N=N, q1=q1, q2=q2, symdiff=d, per_site=d / N))
        print(f"   min |O({q1}) ^ O({q2})| = {d} = {d / N:.4f} N", flush=True)
    if Lx == 8:
        np.save(os.path.join(os.path.dirname(__file__), "..", "data", "family_gs_8x4.npy"),
                np.array([res[q][1][0] for q in (1.02, 1.075, 1.15)]))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data", "family_symdiff.json"), "w"), indent=1)
