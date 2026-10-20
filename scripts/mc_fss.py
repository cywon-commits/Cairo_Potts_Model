"""Finite-size scaling of the PT data: Binder crossings, 2 beta/nu from
<m^2>(Tc) ~ L^(-2 beta/nu), 1/nu from dU_m/dT|Tc ~ L^(1/nu), C_max(L)."""
import glob, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from cairo_potts.analysis import jackknife, crossing

out = {}
for q in (0.4, 1.0, 1.2, 1.4):
    fns = sorted(glob.glob(os.path.join(ROOT, "data", "mc", f"pt_p1_q{q}_L*.npz")),
                 key=lambda f: int(f.split("_L")[-1][:-4]))
    rs = [jackknife(f) for f in fns]
    Ls = np.array([r["L"] for r in rs])
    res = dict(L=Ls.tolist(), trips=[float(r["trips"].sum()) for r in rs],
               acc_min=[float(r["acc"].min()) for r in rs], R=[len(r["T"]) for r in rs])
    cr = []
    for a, b in zip(rs[:-1], rs[1:]):
        for key in ("U_m", "U_Nem" if q == 1.0 else "U_P"):
            T0, y0, eT, ey = crossing(a, b, key)
            cr.append(dict(L1=a["L"], L2=b["L"], key=key, T=T0, dT=eT, U=y0, dU=ey))
    res["crossings"] = cr
    Tc = np.mean([c["T"] for c in cr if c["key"] == "U_m" and c["L1"] >= 24])
    res["Tc_used"] = Tc

    def at(r, key, T, jk=None):
        y = r["val"][key] if jk is None else r["jk"][jk][key]
        return np.interp(T, r["T"], y)

    def slope(r, key, T, jk=None):
        y = r["val"][key] if jk is None else r["jk"][jk][key]
        return np.interp(T, r["T"], np.gradient(y, r["T"]))

    def fit(vals):
        return np.polyfit(np.log(Ls), np.log(vals), 1)[0]

    m2 = np.array([at(r, "m2", Tc) for r in rs])
    du = np.array([-slope(r, "U_m", Tc) for r in rs])
    nj = min(len(r["jk"]) for r in rs)
    e_b = np.std([fit([at(r, "m2", Tc, j) for r in rs]) for j in range(nj)]) * np.sqrt(nj - 1)
    e_n = np.std([fit([-slope(r, "U_m", Tc, j) for r in rs]) for j in range(nj)]) * np.sqrt(nj - 1)
    res["two_beta_over_nu"] = (-fit(m2), e_b)
    res["inv_nu"] = (fit(du), e_n)
    # same, largest three sizes only
    res["two_beta_over_nu_L24up"] = -np.polyfit(np.log(Ls[2:]), np.log(m2[2:]), 1)[0]
    res["inv_nu_L24up"] = np.polyfit(np.log(Ls[2:]), np.log(du[2:]), 1)[0]
    res["Cmax"] = [(r["L"], float(r["val"]["C"].max()), float(r["T"][np.argmax(r["val"]["C"])])) for r in rs]
    out[str(q)] = res
    print(f"q={q}: Ls={Ls.tolist()} R={res['R']} trips={res['trips']} acc_min={np.round(res['acc_min'],2).tolist()}")
    for c in cr:
        print(f"   {c['key']:5s} ({c['L1']},{c['L2']}) T*={c['T']:.4f}({c['dT']*1e4:.0f}) U*={c['U']:.3f}({c['dU']*1e3:.0f})")
    print(f"   Tc(used)={Tc:.4f}  2beta/nu={res['two_beta_over_nu'][0]:.3f}({res['two_beta_over_nu'][1]:.3f}) [L>=24: {res['two_beta_over_nu_L24up']:.3f}]"
          f"  1/nu={res['inv_nu'][0]:.3f}({res['inv_nu'][1]:.3f}) [L>=24: {res['inv_nu_L24up']:.3f}]")
    print("   Cmax:", [(L, round(c, 3), round(T, 4)) for L, c, T in res["Cmax"]])
json.dump(out, open(os.path.join(ROOT, "data", "mc_fss_p1.json"), "w"), indent=1, default=float)
