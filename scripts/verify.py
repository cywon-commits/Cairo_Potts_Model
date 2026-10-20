"""Verification suite (document section 'Verification criteria').

 V1  T -> 0 limits of n3 at p = 1 from the exact transfer matrix.
 V2  Free energy: dense T2 (L = 4, 6) vs. the plaquette-by-plaquette Arnoldi
     implementation used for L >= 8; torus Z(Lx x Ly) = Z(Ly x Lx).
 V3  Explicit-B Cairo model vs. B-integrated plaquette model:
     (a) exhaustive enumeration of the explicit model on a 2 x 2 torus
         vs. Tr T2 of the plaquette model, L = 2;
     (b) explicit-B heat-bath MC on a 6 x 6 torus vs. the exact torus
         partition function Tr T2^3 (L = 6) of the plaquette model;
     (c) plaquette-model PT MC on the same torus vs. the same exact value.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.tm import CylinderTM, free_energy, n3_density
from cairo_potts.explicit import exact_enumeration, explicit_mc
from cairo_potts.mc import PTRun, OBS

report = {}


def torus_lnZ(Lx, L, T, p, q):
    t = CylinderTM(L, 1 / T, p, q)
    M = t.dense_T2()
    ev = np.linalg.eigvals(M)
    return np.log(np.sum(ev ** (Lx // 2)).real) - (1 / T) * t.e0 * L * Lx


def torus_obs(Lx, L, T, p, q, h=1e-5):
    N = 3 * Lx * L
    b = 1 / T
    lz = lambda bb, qq: torus_lnZ(Lx, L, 1 / bb, p, qq)
    E = -(lz(b + h, q) - lz(b - h, q)) / (2 * h) / N
    n3 = -(lz(b, q + h) - lz(b, q - h)) / (2 * h) / (b * N)
    return E, n3


# V1
v1 = {}
for q in (0.5, 0.8, 1.2, 1.4, 1.6, 2.0):
    v1[q] = dict(n3=n3_density(6, 0.02, 1, q, h=1e-5))
report["V1_n3_T0.02_L6"] = v1
print("V1", json.dumps(v1))

# V2
v2 = []
for L in (4, 6):
    for (T, q) in ((0.3, 1.2), (0.14, 1.2), (0.08, 0.8), (0.2, 1.5)):
        t = CylinderTM(L, 1 / T, 1, q)
        ev_d = t.spectrum(k=2)
        ev_a = t.spectrum(k=2, dense_max=0)
        f_d = -np.log(abs(ev_d[0])) / (6 * L / T) + t.e0 / 3
        f_a = -np.log(abs(ev_a[0])) / (6 * L / T) + t.e0 / 3
        v2.append(dict(L=L, T=T, q=q, f_dense=f_d, f_arnoldi=f_a, diff=abs(f_d - f_a)))
lzA = torus_lnZ(6, 4, 0.14, 1, 1.2); lzB = torus_lnZ(4, 6, 0.14, 1, 1.2)
report["V2_free_energy"] = v2
report["V2_torus_rotation"] = dict(lnZ_6x4=lzA, lnZ_4x6=lzB)
print("V2", json.dumps(v2), lzA, lzB)

# V3a
v3a = []
for T in (0.2, 0.5):
    ex = exact_enumeration(2, 2, T, 1, 1.2)
    lz = torus_lnZ(2, 2, T, 1, 1.2)
    v3a.append(dict(T=T, lnZ_explicit=ex["lnZ"], lnZ_plaquette=lz))
report["V3a_2x2_enumeration"] = v3a
print("V3a", json.dumps(v3a))

# V3b/c
v3 = []
for T in (0.3, 0.2, 0.15):
    Eex, n3ex = torus_obs(6, 6, T, 1, 1.2)
    E, n3 = explicit_mc(6, 6, T, 1, 1.2, n_sweeps=400000, seed=3)
    run = PTRun(6, [T], 1, 1.2, seed=5)
    bins = run.run(5000, 50, 8000)[:, 0, :]
    Ep, n3p = bins[:, OBS.index("E")], bins[:, OBS.index("n3")]
    err = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    v3.append(dict(T=T, E_exact=Eex, n3_exact=n3ex,
                   E_explicitMC=E.mean(), E_explicitMC_err=err(E),
                   n3_explicitMC=n3.mean(), n3_explicitMC_err=err(n3),
                   E_plaqMC=Ep.mean(), E_plaqMC_err=err(Ep),
                   n3_plaqMC=n3p.mean(), n3_plaqMC_err=err(n3p)))
    print("V3", json.dumps(v3[-1]), flush=True)
report["V3bc_6x6_torus"] = v3
json.dump(report, open(os.path.join(os.path.dirname(__file__), "..", "data", "verification.json"), "w"), indent=1)
