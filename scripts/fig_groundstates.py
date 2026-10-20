"""Ground-state configurations on a 4 x 4 A torus, drawn on the Cairo lattice
(B sites shifted toward their partner to show the pentagons)."""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cairo_potts.groundstate import ground_states, restore_B

COL = ["#f2f2f2", "#3b7dd8", "#d8483b"]   # colour 1 (cost 0), 2 (p), 3 (q)
D = 0.22


def draw(ax, a, B, title, L=4):
    pos = {}
    for x in range(L + 1):
        for y in range(L + 1):
            pos[("A", x, y)] = (x, y)
    for (x, y), (k, c, cp, _) in B.items():
        if k == "H":   # bottom edge B (c) <-> top edge B (c')
            pb, pt = (x + .5, y + D), (x + .5, y + 1 - D)
            ends = [((x, y), (x + 1, y), pb, c), ((x, y + 1), (x + 1, y + 1), pt, cp)]
        else:          # left edge B (c) <-> right edge B (c')
            pb, pt = (x + D, y + .5), (x + 1 - D, y + .5)
            ends = [((x, y), (x, y + 1), pb, c), ((x + 1, y), (x + 1, y + 1), pt, cp)]
        ax.plot([pb[0], pt[0]], [pb[1], pt[1]], "k-", lw=1, zorder=1)
        for e1, e2, pb_, col in ends:
            for e in (e1, e2):
                ax.plot([pb_[0], e[0]], [pb_[1], e[1]], "k-", lw=1, zorder=1)
            ax.scatter(*pb_, s=90, c=COL[col], edgecolors="k", zorder=3, marker="o")
    for x in range(L + 1):
        for y in range(L + 1):
            ax.scatter(x, y, s=170, c=COL[a[x % L, y % L]], edgecolors="k", zorder=3, marker="s")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal"); ax.axis("off")


cases = [((1, 1.2), "crystal, p<q: colour 3 on one A checkerboard"),
         ((1, 0.8), "crystal, q<p: colour 2 on one A checkerboard"),
         ((1, 1.0), "mirror line p=q: colour-1 lines (one example)"),
         ((1, 1.6), "M phase (unique)")]
fig, axs = plt.subplots(1, 4, figsize=(15, 4.3))
for ax, ((p, q), t) in zip(axs, cases):
    e, confs, tabs = ground_states(4, 4, p, q)
    a = confs[0] if (p, q) != (1, 1.0) else confs[len(confs) // 3]
    draw(ax, a, restore_B(a, tabs), f"(p,q)=({p},{q})  E/N={e:.4f}  #GS(4x4)={len(confs)}\n{t}")
for c, lab in zip(COL, ["colour 1 (cost 0)", "colour 2 (cost p)", "colour 3 (cost q)"]):
    axs[0].scatter([], [], c=c, edgecolors="k", label=lab)
fig.legend(loc="lower center", ncol=3, frameon=False)
fig.text(0.5, 0.92, "squares: A (degree 4), circles: B (degree 3)", ha="center", fontsize=9)
plt.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "ground_states.png"), dpi=150, bbox_inches="tight")
