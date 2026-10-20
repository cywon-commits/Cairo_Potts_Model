"""Figures: (q,T) phase diagram at p = 1, TM scaled gaps, MC Binder
cumulants / specific heat, TM thermodynamics."""
import glob, json, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from cairo_potts.analysis import jackknife, crossing

FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data")


def load_scan(*names):
    D = collections.defaultdict(dict)
    for n in names:
        fn = os.path.join(DATA, n)
        if not os.path.exists(fn):
            continue
        for l in open(fn):
            r = json.loads(l)
            if "error" in r:
                continue
            x = np.array(r["x"]); s = np.array(r["sign"]); im = np.abs(np.array(r["mu_im"][1:]))
            odd = x[(s < 0) & (im < 1e-9)]; even = x[(s > 0) & (im < 1e-9)]
            D[(r["q"], r["L"])][r["T"]] = dict(o1=odd[0] if len(odd) else np.nan,
                                               o2=odd[1] if len(odd) > 1 else np.nan,
                                               e1=even[0] if len(even) else np.nan)
    return D


def cross(D, q, L1, L2, key, xmin=0.003):
    """Highest-T crossing with x above xmin (ignore ordered-phase noise)."""
    a, b = D[(q, L1)], D[(q, L2)]
    Ts = sorted(set(a) & set(b))
    res = []
    for T0, T1 in zip(Ts[:-1], Ts[1:]):
        d0 = a[T0][key] - b[T0][key]; d1 = a[T1][key] - b[T1][key]
        if np.isfinite(d0) and np.isfinite(d1) and d0 * d1 < 0 and d0 > 0:
            Tc = T0 - d0 * (T1 - T0) / (d1 - d0)
            xv = np.interp(Tc, [T0, T1], [b[T0][key], b[T1][key]])
            if xv > xmin:
                res.append((Tc, xv))
    return res


D = load_scan("tm_scan_p1.jsonl", "tm_scan_p1_edge.jsonl", "tm_scan_p1_mirror.jsonl")
qs = sorted({k[0] for k in D})

# ---------------------------------------------------------------- phase diagram
tc1, tc2 = [], []
for q in qs:
    Ls = sorted(L for (qq, L) in D if qq == q)
    if 10 not in Ls or 8 not in Ls:
        continue
    c = cross(D, q, 8, 10, "o1")
    if c:
        tc1.append((q, max(c)[0], max(c)[1]))
    c2 = cross(D, q, 8, 10, "o2")
    # separate crystal <-> line boundary only where it lies below T_c1;
    # for |q - p| >~ 0.15 the two transitions merge
    if c2 and 0.85 < q < 1.15 and c and max(c2)[0] < max(c)[0] - 0.005:
        tc2.append((q, max(c2)[0]))
tc1 = np.array(tc1)
tc2 = np.array(sorted(tc2 + [(1.0, 0.0)]))  # T=0 extensive degeneracy at p = q
fo = [r for r in json.load(open(os.path.join(DATA, "first_order_line_p1.json"))) if r["L"] == 10]
fo = np.array([(r["qc"], r["T"], r["n3_lo"] - r["n3_hi"]) for r in fo if np.isfinite(r["qc"])])

mc = {}
for q in (0.4, 1.0, 1.2, 1.4):
    fns = sorted(glob.glob(os.path.join(DATA, "mc", f"pt_p1_q{q}_L*.npz")),
                 key=lambda f: int(f.split("_L")[-1][:-4]))
    rs = [jackknife(f) for f in fns]
    rs = [r for r in rs if r["nbins"] >= 20]
    if len(rs) >= 2:
        T0, _, eT, _ = crossing(rs[-2], rs[-1], "U_m")
        mc[q] = (T0, eT, rs[-2]["L"], rs[-1]["L"])
json.dump(dict(tc1=tc1.tolist(), tc2=tc2.tolist(), first_order=fo.tolist(),
               mc={str(k): v for k, v in mc.items()}),
          open(os.path.join(DATA, "phase_diagram_p1.json"), "w"), indent=1)

fig, ax = plt.subplots(figsize=(7.5, 5))
m = tc1[:, 0] <= 1.485
ax.plot(tc1[m, 0], tc1[m, 1], "o-", c="#1f5fae", label="continuous (Ashkin–Teller, c=1): TM crossing L=8/10")
o = np.argsort(fo[:, 1])
ax.plot(np.r_[fo[o, 0]], np.r_[fo[o, 1]], "s-", c="#c0392b", ms=4, label="first order (n$_3$ jump): TM L=10")
if len(tc2):
    ax.plot(tc2[:, 0], tc2[:, 1], "^--", c="#7d3c98", label="crystal ↔ colour-1 line phase (TM, L=8/10)")
for q, (T0, eT, L1, L2) in mc.items():
    ax.errorbar(q, T0, yerr=eT, fmt="*", c="k", ms=11, zorder=5)
ax.plot([], [], "*", c="k", ms=11, label="PT Monte Carlo: Binder $U_m$ crossing (two largest L)")
ax.axvline(1.0, c="grey", lw=.6, ls=":")
ax.text(1.0, 0.012, "p=q", ha="center", fontsize=8, color="grey")
ax.text(0.35, 0.10, "crystal\n(4 states, colour 1\non one of 4 A sub-sublattices)", fontsize=8, ha="center")
ax.text(1.27, 0.06, "crystal", fontsize=8, ha="center")
ax.text(1.0, 0.075, "line\nphase", fontsize=8, ha="center", color="#7d3c98")
ax.text(1.53, 0.08, "M /\nparamagnet", fontsize=8, ha="center")
ax.text(0.65, 0.235, "paramagnet", fontsize=9)
ax.set_xlim(-0.02, 1.6); ax.set_ylim(0, 0.3)
ax.set_xlabel("q = D$_3$  (p = D$_2$ = 1, J = 1)"); ax.set_ylabel("T / J")
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Cairo AF 3-state Potts, p = 1: finite-temperature phase diagram", fontsize=10)
plt.savefig(os.path.join(FIG, "phase_diagram_p1.png"), dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- TM gaps
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for ax, q in zip(axs, (0.4, 1.4, 1.49)):
    for L, c in zip((4, 6, 8, 10), ("#bbb", "#888", "#444", "#000")):
        d = D.get((q, L), {})
        Ts = sorted(d)
        for key, ls in (("o1", "-"), ("o2", "--"), ("e1", ":")):
            ax.plot(Ts, [d[T][key] for T in Ts], ls, c=c, label=f"L={L} {key}" if key == "o1" else None)
    ax.axhline(1 / 8, c="r", lw=.7); ax.text(Ts[0] if Ts else 0, 0.13, "1/8", color="r", fontsize=8)
    ax.set_ylim(0, 0.6); ax.set_xlabel("T"); ax.set_title(f"q={q}: scaled gaps x=L ln|μ0/μi|/2π\n(— odd 1, -- odd 2, ··· even 1)", fontsize=9)
axs[0].legend(fontsize=7); axs[0].set_ylabel("x")
plt.savefig(os.path.join(FIG, "tm_gaps.png"), dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------- MC
fig, axs = plt.subplots(3, 4, figsize=(16, 10))
for j, q in enumerate((0.4, 1.0, 1.2, 1.4)):
    fns = sorted(glob.glob(os.path.join(DATA, "mc", f"pt_p1_q{q}_L*.npz")),
                 key=lambda f: int(f.split("_L")[-1][:-4]))
    for fn in fns:
        r = jackknife(fn)
        if r["nbins"] < 20:
            continue
        for i, (key, lab) in enumerate((("U_m", "$U_m$ (clock)"), ("C", "C per site"), ("phi4", "⟨m⁴cos4θ⟩/⟨m⁴⟩"))):
            axs[i, j].errorbar(r["T"], r["val"][key], yerr=r["err"][key], fmt=".-", label=f"L={r['L']}")
            axs[i, j].set_ylabel(lab)
    axs[0, j].set_title(f"q={q}")
    axs[2, j].set_xlabel("T")
axs[0, 0].legend(fontsize=7)
plt.savefig(os.path.join(FIG, "mc_binder_C_phi4.png"), dpi=130, bbox_inches="tight")

# ---------------------------------------------------------------- thermodynamics
fn = os.path.join(DATA, "tm_thermo_p1_L8.json")
if os.path.exists(fn):
    rows = json.load(open(fn))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for q in sorted({r["q"] for r in rows}):
        rr = sorted([r for r in rows if r["q"] == q], key=lambda r: r["T"])
        T = [r["T"] for r in rr]
        for ax, k in zip(axs, ("s", "c", "n3")):
            ax.plot(T, [r[k] for r in rr], label=f"q={q}")
    axs[0].axhline(np.log(2) / 4, c="k", lw=.6, ls=":"); axs[0].text(0.03, np.log(2) / 4 + .005, "ln2/4", fontsize=8)
    for ax, t in zip(axs, ("entropy per site", "specific heat per site", "n3")):
        ax.set_title(t + " (exact TM, L=8 cylinder)", fontsize=9); ax.set_xlabel("T")
    axs[2].legend(fontsize=7)
    plt.savefig(os.path.join(FIG, "tm_thermodynamics.png"), dpi=140, bbox_inches="tight")
print(json.dumps(dict(tc1=tc1.tolist(), tc2=tc2.tolist(), mc={str(k): v for k, v in mc.items()}), indent=0))
