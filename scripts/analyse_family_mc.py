"""Request 2 MC analysis: n3(T), chi3 = N Var(n3) vs L (first order: grows
~ N at coexistence; crossover: L-independent), histogram modality, and a
figure with the T = 0 kink structure and the finite-T crossover."""
import glob, json, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")


def modes(h, edges, frac=0.05):
    """Number of separated local maxima above frac * max after light smoothing."""
    if h.sum() == 0:
        return 0
    k = np.ones(5) / 5
    s = np.convolve(h, k, mode="same")
    pk = [i for i in range(1, len(s) - 1) if s[i] > s[i - 1] and s[i] >= s[i + 1] and s[i] > frac * s.max()]
    merged = []
    for i in pk:
        if not merged or i - merged[-1] > 5 and s[merged[-1]:i].min() < 0.5 * min(s[merged[-1]], s[i]):
            merged.append(i)
    return len(merged)


SUB = sys.argv[1] if len(sys.argv) > 1 else "mc_family"
res = collections.defaultdict(dict)
for fn in sorted(glob.glob(os.path.join(ROOT, "data", "mc_family", "fam_q*_L16.npz"))
                 + glob.glob(os.path.join(ROOT, "data", SUB, "fam_q*_L*.npz"))):
    d = np.load(fn); meta = json.loads(str(d["meta"]))
    L, q = meta["L"], meta["q"]; N = 3 * L * L
    b = d["bins"]; b = b[np.abs(b).sum(axis=(1, 2)) > 0]
    obs = meta["obs"]; i3, i32 = obs.index("n3"), obs.index("n32")
    n3 = b[:, :, i3].mean(0); e3 = b[:, :, i3].std(0, ddof=1) / np.sqrt(len(b))
    chi = N * (b[:, :, i32].mean(0) - n3 ** 2)
    # jackknife error of chi
    jk = np.array([N * (np.delete(b[:, :, i32], j, 0).mean(0) - np.delete(b[:, :, i3], j, 0).mean(0) ** 2)
                   for j in range(len(b))])
    echi = np.sqrt((len(b) - 1) * jk.var(0))
    nm = [modes(d["hn3"][i], d["edges_n3"]) for i in range(len(d["Ts"]))]
    res[q][L] = dict(T=d["Ts"].tolist(), n3=n3.tolist(), n3_err=e3.tolist(), chi3=chi.tolist(),
                     chi3_err=echi.tolist(), n3_modes=nm, acc=d["acc"].tolist(), nbins=len(b))
    print(f"q={q} L={L} bins={len(b)} acc_min={d['acc'].min():.2f}")
    for i, T in enumerate(d["Ts"]):
        print(f"   T={T:.4f} n3={n3[i]:.5f}({e3[i]*1e5:.0f}) chi3=N*Var(n3)={chi[i]:.4f}({echi[i]:.4f}) modes={nm[i]}")
json.dump(res, open(os.path.join(ROOT, "data", f"family_mc_summary_{SUB}.json"), "w"), indent=1)

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
k = json.load(open(os.path.join(ROOT, "data", "family_x34_p0.1_T0_kinks.json")))["8"]
q = np.array(k["q"]); E = np.array(k["E0"])
axs[0].step(0.5 * (q[1:] + q[:-1]), np.diff(E) / np.diff(q), where="mid", c="k", label="T = 0, cylinder L=8 (bulk)")
axs[0].axhline(1 / 24, c="#c0392b", ls=":", lw=.8)
axs[0].plot([0.15, 4 / 3, 4 / 3, 1.8], [1 / 6, 1 / 6, 1 / 24, 1 / 24], c="#c0392b", ls="--", lw=1,
            label="T = 0, 4x4 torus (commensurability artifact)")
rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "family_x34_finiteT.jsonl"))]
low = [json.loads(l) for l in open(os.path.join(ROOT, "data", "family_x34_lowT.jsonl"))]
# low T from the fine scan (L = 6; the L = 8, T = 0.02 points near q = 1.07 are numerically unstable)
for T, c, src, L in ((0.015, "#0b3d7a", low, 6), (0.03, "#1f5fae", low, 6), (0.06, "#3d8ad1", rows, 8),
                     (0.12, "#7fb3e6", rows, 8), (0.2, "#bcd7f2", rows, 8)):
    rr = sorted([r for r in src if r.get("L") == L and r["T"] == T and "n3" in r], key=lambda r: r["q"])
    axs[0].plot([r["q"] for r in rr], [r["n3"] for r in rr], c=c, label=f"TM L={L}, T={T}")
axs[0].set_xlim(0.9, 1.45); axs[0].set_ylim(0, 0.2); axs[0].set_xlabel("q"); axs[0].set_ylabel("f_exp / n3")
axs[0].legend(fontsize=7); axs[0].set_title("x = 3/4 stripes, p = 0.1: kinks at q = 1.05, 1.10", fontsize=9)
for j, qq in enumerate(sorted(res)):
    for L in sorted(res[qq]):
        r = res[qq][L]
        axs[1 + j].errorbar(r["T"], r["chi3"], yerr=r["chi3_err"], fmt=".-", label=f"L={L}")
    axs[1 + j].set_xscale("log"); axs[1 + j].set_xlabel("T"); axs[1 + j].set_ylabel("N Var(n3)")
    axs[1 + j].set_title(f"q = {qq}: n3 fluctuation (first order would grow ∝ N)", fontsize=9)
    axs[1 + j].legend(fontsize=7)
plt.savefig(os.path.join(ROOT, "figures", "family_x34_request2.png"), dpi=140, bbox_inches="tight")
