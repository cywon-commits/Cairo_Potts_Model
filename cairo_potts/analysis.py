"""Jackknife analysis of PT bins."""
import json
import numpy as np

from .mc import OBS

I = {k: i for i, k in enumerate(OBS)}


def load(fn):
    d = np.load(fn)
    meta = json.loads(str(d["meta"]))
    bins = d["bins"]
    nb = int((np.abs(bins).sum(axis=(1, 2)) > 0).sum())  # completed bins
    return dict(bins=bins[:nb], Ts=d["Ts"], acc=d["acc"], trips=d["trips"],
                L=meta["L"], q=meta["q"], p=meta["p"], meta=meta)


def derived(mean, L, Ts):
    """mean: (R, nobs) averages -> dict of derived observables per T."""
    g = lambda k: mean[:, I[k]]
    N = 3 * L * L
    NA = L * L
    out = {}
    out["E"] = g("E")
    out["C"] = N * (g("E2") - g("E") ** 2 + g("Evar")) / Ts ** 2
    out["n3"] = g("n3")
    out["n2"] = g("n2")
    out["nb"] = g("nb")
    out["m2"] = g("m2")
    out["U_m"] = 1 - g("m4") / (2 * g("m2") ** 2)
    out["chi_m"] = NA * g("m2") / Ts
    out["chi_m_conn"] = NA * (g("m2") - g("absm") ** 2) / Ts
    out["P2"] = g("P2")
    out["U_P"] = 1 - g("P4") / (3 * g("P2") ** 2)
    out["Nem2"] = g("Nem2")
    out["U_Nem"] = 1 - g("Nem4") / (3 * g("Nem2") ** 2)
    out["phi4"] = g("c4m4") / g("m4")
    out["sig2"] = g("sig2"); out["tau2"] = g("tau2")
    return out


def jackknife(fn, nblock=20):
    d = load(fn)
    bins, Ts, L = d["bins"], d["Ts"], d["L"]
    nb = len(bins)
    nblock = min(nblock, nb)
    blocks = np.array_split(np.arange(nb), nblock)
    full = derived(bins.mean(0), L, Ts)
    jk = []
    for b in blocks:
        mask = np.ones(nb, bool); mask[b] = False
        jk.append(derived(bins[mask].mean(0), L, Ts))
    err = {k: np.sqrt((nblock - 1) / nblock * np.sum([(j[k] - np.mean([jj[k] for jj in jk], 0)) ** 2 for j in jk], 0))
           for k in full}
    return dict(T=Ts, L=L, q=d["q"], p=d["p"], val=full, err=err, nbins=nb,
                acc=d["acc"], trips=d["trips"], jk=jk)


def crossing(r1, r2, key, Tlo=None, Thi=None):
    """Crossing of key between two runs on (possibly different) T grids, with
    a jackknife error (blocks paired by index)."""
    def cross(T1, y1, T2, y2):
        Tg = np.linspace(max(T1.min(), T2.min()), min(T1.max(), T2.max()), 2000)
        if Tlo is not None:
            Tg = Tg[(Tg >= Tlo) & (Tg <= Thi)]
        d = np.interp(Tg, T1, y1) - np.interp(Tg, T2, y2)
        s = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if len(s) == 0:
            return np.nan, np.nan
        i = s[0]
        T0 = Tg[i] - d[i] * (Tg[i + 1] - Tg[i]) / (d[i + 1] - d[i])
        return T0, np.interp(T0, T1, y1)
    T0, y0 = cross(r1["T"], r1["val"][key], r2["T"], r2["val"][key])
    n = min(len(r1["jk"]), len(r2["jk"]))
    js = np.array([cross(r1["T"], r1["jk"][i][key], r2["T"], r2["jk"][i][key]) for i in range(n)])
    js = js[np.isfinite(js).all(1)]
    if len(js) < 2:
        return T0, y0, np.nan, np.nan
    e = np.sqrt((len(js) - 1) / len(js) * ((js - js.mean(0)) ** 2).sum(0))
    return T0, y0, e[0], e[1]
