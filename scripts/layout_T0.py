"""Request 4, T = 0 part, for column layouts 'n:S' (paired columns x mod n in S):
ground-state x-period (smallest torus length Lx, multiple of P, whose energy
equals the cylinder energy), f_exp = dE0/dq, and residual entropy per site
from exact (min, count) ground-state counting on Lx x Ly tori."""
import json, os, sys
from math import gcd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm_family import (parse_layout, ground_energy_family, torus_ground_energy,
                                   torus_ground_count)

p = 0.1
out = []
for spec in sys.argv[2:]:
    sel, n, x = parse_layout(spec)
    P = 2 * n // gcd(2, n)
    for q in [float(v) for v in sys.argv[1].split(",")]:
        Ecyl = ground_energy_family(4, p, q, sel, n, iters=60)
        Ecyl6 = ground_energy_family(6, p, q, sel, n, iters=60)
        h = 1e-4
        fexp = (ground_energy_family(4, p, q + h, sel, n, iters=60)
                - ground_energy_family(4, p, q - h, sel, n, iters=60)) / (2 * h)
        period = None
        for m in range(1, 9):
            Lx = m * P
            if abs(torus_ground_energy(Lx, 4, p, q, sel, n) - Ecyl) < 1e-10:
                period = Lx; break
        row = dict(layout=spec, x=x, q=q, P=P, E0=Ecyl, E0_L6=Ecyl6, f_exp=fexp, gs_period_x=period)
        if period:
            lg = {}
            for Ly in (2, 4):
                Lxs = [period * k for k in (2, 3)] if period * 3 <= 48 else [period, 2 * period]
                c = [torus_ground_count(Lx, Ly, p, q, sel, n)[1] for Lx in Lxs]
                lg[Ly] = (c[1] - c[0]) / (3 * Ly * (Lxs[1] - Lxs[0]))
            row["s0_Ly2"], row["s0_Ly4"] = lg[2], lg[4]
            row["fexp_times_N_torus"] = fexp * 3 * period * 4
        out.append(row)
        print(json.dumps(row), flush=True)
fn = os.path.join(os.path.dirname(__file__), "..", "data", "layout_T0.json")
old = json.load(open(fn)) if os.path.exists(fn) else []
json.dump(old + out, open(fn, "w"), indent=1)
