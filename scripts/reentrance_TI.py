"""Request 5 MC check: coexistence temperature of the M and Potts phases at
p = 0.3, q = 1.87 from thermodynamic integration along the two metastable
MC branches (hysteresis): beta f(beta) = beta0 f(beta0) + int E d beta,
starting at T0 = 0.015 where both phases have negligible entropy (TM:
s_Potts <~ 1e-4, s_M ~ 1e-7, so T0 s0 < 1e-5 << the free-energy
differences of ~1e-3 involved); f(T0) = E(T0).
Branches: random start -> Potts (metastable at low T), M start -> M
(metastable at high T); only replicas still in their phase are used.
(A first attempt from T0 = 0.05 assumed s_Potts ~ 0 there, which is wrong:
s_Potts(0.05) ~ 0.03.)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.analysis import jackknife

D = os.path.join(os.path.dirname(__file__), "..", "data", "mc_p0.3")


def branch(fn, sel):
    r = jackknife(fn)
    T = r["T"]; E = r["val"]["E"]; n3 = r["val"]["n3"]
    m = sel(T, n3)
    return T[m], E[m], np.array([j["E"][m] for j in r["jk"]])


def betaf(T, E):
    b = 1 / T                       # decreasing beta
    bf = np.empty_like(T)
    bf[0] = b[0] * E[0]             # f(T0) ~ E(T0)
    for i in range(1, len(T)):
        bf[i] = bf[i - 1] + 0.5 * (E[i] + E[i - 1]) * (b[i] - b[i - 1])
    return bf * T                   # f(T)


out = {}
for L, fp, fm in ((16, "TI_potts_L16.npz", "TI_M_L16.npz"),
                  (32, "TI_potts_L32.npz", "TI_M_L32.npz")):
    if fp is None:
        continue
    Tp, Ep, jp = branch(os.path.join(D, fp), lambda T, n: (T <= 0.115) & (n > 0.05))
    Tm, Em, jm = branch(os.path.join(D, fm), lambda T, n: (n < 0.01))
    # keep only the contiguous low-T part of each branch
    Tp, Ep, jp = Tp[:np.argmax(np.diff(np.r_[Tp, 9]) > 0.006) + 1], Ep, jp
    Ep, jp = Ep[:len(Tp)], jp[:, :len(Tp)]
    Tg = Tm[Tm <= Tp.max()]
    def cross(Ep_, Em_):
        fP = np.interp(Tg, Tp, betaf(Tp, Ep_)); fM = betaf(Tm, Em_)[:len(Tg)]
        d = fP - fM
        i = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if not len(i):
            return np.nan, d
        i = i[0]
        return Tg[i] - d[i] * (Tg[i + 1] - Tg[i]) / (d[i + 1] - d[i]), d
    Tx, d = cross(Ep, Em)
    js = [cross(jp[k], jm[k])[0] for k in range(min(len(jp), len(jm)))]
    err = np.sqrt((len(js) - 1) * np.nanvar(js))
    out[L] = dict(T_coex=Tx, err=err, T=Tg.tolist(), fP_minus_fM=d.tolist())
    print(f"L={L}: f_Potts - f_M on T grid: " + " ".join(f"{t:.4f}:{x:+.2e}" for t, x in zip(Tg, d)))
    print(f"L={L}: coexistence T_x = {Tx:.4f} +- {err:.4f}")
json.dump({str(k): v for k, v in out.items()}, open(os.path.join(D, "reentrance_TI.json"), "w"), indent=1)
