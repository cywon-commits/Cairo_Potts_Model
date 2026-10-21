"""Request 5 figure: delta_q(T) = q_c(T) - q_0(p) for several p, and p*(T)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
d = json.load(open(os.path.join(ROOT, "data", "p_star.json")))
ps = json.load(open(os.path.join(ROOT, "data", "p_star_T.json")))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for p in sorted({r["p"] for r in d}):
    rr = sorted([r for r in d if r["p"] == p], key=lambda r: r["T"])
    axs[0].plot([r["T"] for r in rr], [r["dq"] for r in rr], "o-", label=f"p={p}")
axs[0].axhline(0, c="k", lw=.6); axs[0].set_yscale("symlog", linthresh=1e-6)
axs[0].set_xlabel("T"); axs[0].set_ylabel("q_c(T) - (4-p)/2")
axs[0].set_title("> 0: line bends to larger q (re-entrant);  < 0: to smaller q", fontsize=9)
axs[0].legend(fontsize=7)
T = np.array([r["T"] for r in ps]); P = np.array([r["p_star"] for r in ps])
m = T <= 0.06
c = np.polyfit(T[m], P[m], 1)
axs[1].plot(T, P, "o", c="#c0392b", label="p*(T): zero of q_c(T) - q_0 (TM L=8)")
tt = np.linspace(0, 0.1, 50)
axs[1].plot(tt, np.polyval(c, tt), "--", c="k", lw=.8, label=f"linear fit T<=0.06: p*(0) = {c[1]:.4f}, slope {c[0]:.2f}")
axs[1].set_xlabel("T"); axs[1].set_ylabel("p*"); axs[1].legend(fontsize=7)
axs[1].set_title("re-entrant (outward) for p < p*(T); p*(T -> 0) = 1 = J", fontsize=9)
plt.savefig(os.path.join(ROOT, "figures", "p_star.png"), dpi=140, bbox_inches="tight")
