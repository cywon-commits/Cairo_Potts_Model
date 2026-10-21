"""Binder cumulant of the template order parameter M (x = 3/4 family):
U = 1 - <M^4> / (3 <M^2>^2), crossings between sizes, jackknife errors."""
import glob, json, os, sys, collections
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")


def load(fn):
    d = np.load(fn); meta = json.loads(str(d["meta"])); obs = meta["obs"]
    b = d["bins"]; b = b[np.abs(b).sum(axis=(1, 2)) > 0]
    return meta, d["Ts"], b, obs, d["acc"]


def U(b, obs):
    m2 = b[:, :, obs.index("M2")].mean(0); m4 = b[:, :, obs.index("M4")].mean(0)
    return 1 - m4 / (3 * m2 ** 2), m2


# window for the crossing search (the U curves coincide deep in the ordered phase)
TLO = float(sys.argv[1]) if len(sys.argv) > 2 else 0.24
THI = float(sys.argv[2]) if len(sys.argv) > 2 else 0.29
SUB = sys.argv[3] if len(sys.argv) > 3 else "mc_family_op"
res = collections.defaultdict(dict)
for fn in sorted(glob.glob(os.path.join(ROOT, "data", SUB, "fam_q*_L*.npz"))):
    meta, Ts, b, obs, acc = load(fn)
    u, m2 = U(b, obs)
    nb = len(b); blocks = np.array_split(np.arange(nb), 20)
    jk = np.array([U(np.delete(b, bl, 0), obs)[0] for bl in blocks])
    eu = np.sqrt((len(blocks) - 1) * jk.var(0))
    res[meta["q"]][meta["L"]] = dict(T=Ts, U=u, eU=eu, m2=m2, jk=jk, acc=acc, nbins=nb)

out = {}
for q, rr in sorted(res.items()):
    Ls = sorted(rr)
    print(f"q={q}: sizes {Ls}, bins {[rr[L]['nbins'] for L in Ls]}, acc_min {[round(float(rr[L]['acc'].min()),2) for L in Ls]}")
    for L1, L2 in zip(Ls[:-1], Ls[1:]):
        a, b2 = rr[L1], rr[L2]
        def cross(u1, u2):
            Tg = np.linspace(TLO, THI, 4000)
            d = np.interp(Tg, a["T"], u1) - np.interp(Tg, b2["T"], u2)
            s = np.where((d[:-1] < 0) & (d[1:] >= 0) | (d[:-1] > 0) & (d[1:] <= 0))[0]
            if not len(s):
                return np.nan, np.nan
            i = s[-1]
            T0 = Tg[i] - d[i] * (Tg[i + 1] - Tg[i]) / (d[i + 1] - d[i])
            return T0, np.interp(T0, a["T"], u1)
        T0, U0 = cross(a["U"], b2["U"])
        js = np.array([cross(a["jk"][j], b2["jk"][j]) for j in range(min(len(a["jk"]), len(b2["jk"])))])
        js = js[np.isfinite(js).all(1)]
        e = np.sqrt((len(js) - 1) * js.var(0)) if len(js) > 1 else [np.nan, np.nan]
        out[f"{q}_{L1}_{L2}"] = dict(T=T0, dT=e[0], U=U0, dU=e[1])
        print(f"   U_M crossing ({L1},{L2}): T*={T0:.4f}({e[0]*1e4:.0f})  U*={U0:.3f}({e[1]*1e3:.0f})")
    for L in Ls:
        r = rr[L]; i = np.argmin(np.abs(r["T"] - 0.25))
        print(f"   L={L}: U(T) at T=0.20,0.25,0.30: {np.round(np.interp([0.2,0.25,0.3], r['T'], r['U']),3).tolist()}")
json.dump(out, open(os.path.join(ROOT, "data", f"family_op_crossings_{SUB}.json"), "w"), indent=1)
