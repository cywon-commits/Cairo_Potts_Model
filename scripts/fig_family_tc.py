"""Figure for the family order-disorder transition (request 3 extension)."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")


def tm_cross(fn_list, n, q):
    rows = []
    for fn in fn_list:
        for l in open(os.path.join(D, fn)):
            r = json.loads(l)
            if "error" not in r and r["n"] == n and abs(r["q"] - q) < 1e-9:
                rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(r["L"], {})[r["T"]] = r["x"][0]
    Ls = sorted(by); out = []
    for L1, L2 in zip(Ls[:-1], Ls[1:]):
        Ts = sorted(set(by[L1]) & set(by[L2]))
        d = np.array([by[L1][T] - by[L2][T] for T in Ts])
        for i in range(len(Ts) - 1):
            if d[i] > 0 and d[i + 1] < 0:
                out.append((0.5 * (L1 + L2), Ts[i] - d[i] * (Ts[i + 1] - Ts[i]) / (d[i + 1] - d[i])))
    return out


mc = {}
for sub in ("mc_family_op", "mc_family_op_q0.5"):
    fn = os.path.join(D, f"family_op_crossings_{sub}.json")
    if os.path.exists(fn):
        for k, v in json.load(open(fn)).items():
            q, L1, L2 = k.split("_"); mc.setdefault(float(q), []).append((int(L1), int(L2), v))

fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
ax = axs[0]
for (n, q, fns, c, lab) in ((4, 1.05, ["family_gap_scan.jsonl", "family_gap_scan_L12.jsonl"], "#c0392b", "x=3/4, q=1.05"),
                            (0, 1.05, ["family_gap_scan.jsonl"], "#1f5fae", "Cairo, q=1.05"),
                            (4, 0.5, ["family_gap_scan_q0.5_fine.jsonl"], "#e67e22", "x=3/4, q=0.5"),
                            (0, 0.5, ["family_gap_scan_q0.5_fine.jsonl"], "#5dade2", "Cairo, q=0.5")):
    cr = tm_cross(fns, n, q)
    if cr:
        ax.plot([1 / l for l, _ in cr], [t for _, t in cr], "o-", c=c, label="TM " + lab)
    for (L1, L2, v) in mc.get(q, []):
        if n == 4 and L1 >= 24:
            ax.errorbar(1 / (0.5 * (L1 + L2)), v["T"], yerr=v["dT"], fmt="*", c=c, ms=12,
                        label=f"MC U_M ({L1},{L2}) " + lab)
ax.set_xlabel("1 / L (mean of pair)"); ax.set_ylabel("crossing temperature")
ax.set_xlim(0, 0.24); ax.legend(fontsize=7); ax.set_title("gap / Binder crossings vs size", fontsize=9)

ax = axs[1]
for sub, q, c0 in (("mc_family_op", 1.05, "Reds"), ("mc_family_op_q0.5", 0.5, "Oranges")):
    for k, fn in enumerate(sorted(glob.glob(os.path.join(D, sub, f"fam_q{q}_L*.npz")),
                                  key=lambda f: int(f.split("_L")[-1][:-4]))):
        d = np.load(fn); meta = json.loads(str(d["meta"])); o = meta["obs"]
        b = d["bins"]; m2 = b[:, :, o.index("M2")].mean(0); m4 = b[:, :, o.index("M4")].mean(0)
        ax.plot(d["Ts"], 1 - m4 / (3 * m2 ** 2), ".-", c=plt.get_cmap(c0)(0.4 + 0.25 * k),
                label=f"q={q} L={meta['L']}")
ax.set_xlabel("T"); ax.set_ylabel("U_M = 1 - <M^4>/3<M^2>^2"); ax.legend(fontsize=7)
ax.set_title("x = 3/4 MC: Binder cumulant of the template order parameter", fontsize=9)

ax = axs[2]
tc = json.load(open(os.path.join(D, "family_tc_summary.json")))
for q, mk in (("0.5", "o-"), ("1.05", "s-"), ("1.1", "^-")):
    xs = sorted(tc[q], key=float)
    ax.plot([float(x) for x in xs], [tc[q][x]["Tc"] for x in xs], mk, label=f"p=0.1, q={q}")
ax.set_xlabel("pairing fraction x"); ax.set_ylabel("T_c"); ax.set_xlim(0.2, 1.05)
ax.legend(fontsize=8); ax.set_title("T_c(x): 0 means no order at any T > 0", fontsize=9)
plt.savefig(os.path.join(ROOT, "figures", "family_tc.png"), dpi=140, bbox_inches="tight")
