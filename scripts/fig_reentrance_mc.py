"""Request 5 MC check figure: f_Potts - f_M along the two metastable MC
branches (thermodynamic integration) at p = 0.3, q = 1.87."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
d = json.load(open(os.path.join(ROOT, "data", "mc_p0.3", "reentrance_TI.json")))
fig, ax = plt.subplots(figsize=(6.2, 4))
for L, c in (("16", "#7fb3e6"), ("32", "#1f5fae")):
    ax.plot(d[L]["T"], d[L]["fP_minus_fM"], "o-", ms=3, c=c, label=f"MC L={L}: T_x = {d[L]['T_coex']:.4f}")
ax.axhline(0, c="k", lw=.6)
ax.axvline(0.0891, c="#c0392b", ls="--", lw=1, label="TM (L=10) coexistence at q=1.87: T = 0.0891")
ax.set_xlabel("T"); ax.set_ylabel("f_Potts - f_M  (per site)")
ax.set_title("p = 0.3, q = 1.87 (> T=0 boundary 1.85): M below T_x, Potts above", fontsize=9)
ax.legend(fontsize=7)
plt.savefig(os.path.join(ROOT, "figures", "reentrance_mc.png"), dpi=140, bbox_inches="tight")
