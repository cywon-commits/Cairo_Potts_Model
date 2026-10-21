"""Experiment 3: phase diagram near the Potts-M boundary at p = 0.3."""
import json, os, sys, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "data")


def crossings(files):
    by = collections.defaultdict(dict)
    for f in files:
        for l in open(os.path.join(D, f)):
            r = json.loads(l)
            if "error" in r:
                continue
            x = np.array(r["x"]); s = np.array(r["sign"])
            odd = x[s < 0]
            by[(r["q"], r["L"])][r["T"]] = odd[0] if len(odd) else np.nan
    up, lo = [], []
    for q in sorted({k[0] for k in by}):
        a, b = by.get((q, 8)), by.get((q, 10))
        if not a or not b:
            continue
        Ts = sorted(set(a) & set(b)); d = np.array([a[T] - b[T] for T in Ts])
        for i in range(len(Ts) - 1):
            if np.isfinite(d[i]) and np.isfinite(d[i + 1]) and d[i] * d[i + 1] < 0 and max(b[Ts[i]], b[Ts[i + 1]]) > 0.01:
                T0 = Ts[i] - d[i] * (Ts[i + 1] - Ts[i]) / (d[i + 1] - d[i])
                (up if d[i] > 0 else lo).append((q, T0))
    return np.array(up), np.array(lo)


up, lo = crossings(["tm_scan_p0.3.jsonl", "tm_scan_p0.3_edge.jsonl", "tm_scan_p0.3_reentrant.jsonl"])
fo = []
for f in ("first_order_line_p0.3_part1.log", "first_order_line_p0.3_part2.log"):
    p = os.path.join(D, f)
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r["L"] == 10 and r.get("n3_lo") is not None and r["n3_lo"] - r["n3_hi"] > 0.02:
                fo.append((r["qc"], r["T"], r["n3_lo"] - r["n3_hi"]))
fo = np.array(sorted(fo, key=lambda t: t[1]))
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(up[:, 0], up[:, 1], "o-", c="#1f5fae", label="continuous (AT), TM crossing L=8/10, ordered below")
if len(lo):
    ax.plot(lo[:, 0], lo[:, 1], "v--", c="#7d3c98", label="re-entrant lower boundary (gap crossing, ordered above)")
ax.plot(fo[:, 0], fo[:, 1], "s-", c="#c0392b", ms=4, label="first order (n3 jump), TM L=10")
ax.axvline(1.85, c="grey", lw=.6, ls=":"); ax.text(1.851, 0.01, "T=0 boundary\np+2q=4", fontsize=7, color="grey")
ax.set_xlim(0.7, 2.02); ax.set_ylim(0, 0.27); ax.set_xlabel("q (p = 0.3)"); ax.set_ylabel("T")
ax.legend(fontsize=7, loc="upper right"); ax.set_title("p = 0.3: ordered (Potts) phase is re-entrant beyond the T = 0 boundary", fontsize=9)
ins = ax.inset_axes([0.1, 0.1, 0.42, 0.5])
ins.plot(up[:, 0], up[:, 1], "o-", c="#1f5fae", ms=3)
if len(lo):
    ins.plot(lo[:, 0], lo[:, 1], "v--", c="#7d3c98", ms=3)
ins.plot(fo[:, 0], fo[:, 1], "s-", c="#c0392b", ms=3); ins.set_xlim(1.83, 1.98); ins.set_ylim(0, 0.2)
ins.axvline(1.85, c="grey", lw=.6, ls=":"); ins.tick_params(labelsize=6)
plt.savefig(os.path.join(ROOT, "figures", "phase_diagram_p0.3.png"), dpi=140, bbox_inches="tight")
json.dump(dict(upper=up.tolist(), lower=lo.tolist(), first_order=fo.tolist()),
          open(os.path.join(D, "phase_diagram_p0.3.json"), "w"), indent=1)
