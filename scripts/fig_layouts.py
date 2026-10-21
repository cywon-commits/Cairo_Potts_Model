"""Request 4 figure: T_c vs pairing fraction x for all column layouts;
filled = ground state doubles the structural period P (ordered), open = not."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
d = json.load(open(os.path.join(ROOT, "data", "layout_tc_summary.json")))["layouts"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)
for ax, q in zip(axs, ("0.5", "1.05")):
    for r in d:
        tc = r["Tc"][q][0]; dbl = r["gs_period"][q] == 2 * r["P"] or r["spec"] == "cairo"
        lab = "Cairo" if r["spec"] == "cairo" else f"({r['w']},{r['g']})"
        ax.plot(r["x"], tc, "o", ms=9, mfc="#1f5fae" if dbl else "white", mec="#1f5fae" if dbl else "#c0392b", mew=1.6)
        ax.annotate(lab, (r["x"], tc), textcoords="offset points", xytext=(5, 4 if tc > 0 else -12), fontsize=7)
    ax.set_xlabel("pairing fraction x"); ax.set_ylabel("T_c"); ax.set_xlim(0.2, 1.06)
    ax.set_title(f"p = 0.1, q = {q}: filled = ground state doubles P (ordered); open = no doubling", fontsize=8.5)
plt.savefig(os.path.join(ROOT, "figures", "layouts_tc.png"), dpi=140, bbox_inches="tight")
