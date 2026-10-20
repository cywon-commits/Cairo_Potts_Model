"""AT exponent table along p = 1 from TM crossings (L = 8/10):
x_sigma ~ odd gaps, x_P = even gap, x_eps = 4 x_P, nu = 1/(2 - x_eps)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("mf", os.path.join(os.path.dirname(__file__), "make_figures.py"))
src = open(spec.origin).read().split("# ---------------------------------------------------------------- phase diagram")[0]
ns = {"__file__": spec.origin}
exec(src, ns)
D = ns["load_scan"]("tm_scan_p1.jsonl", "tm_scan_p1_edge.jsonl")
rows = []
for q in sorted({k[0] for k in D}):
    r = dict(q=q)
    for key in ("o1", "o2", "e1"):
        c = ns["cross"](D, q, 8, 10, key)
        r[key] = max(c) if c else None
    if r["o1"] and r["e1"]:
        xe = 4 * r["e1"][1]
        r["x_eps"] = xe; r["nu"] = 1 / (2 - xe) if xe < 2 else None
        print(f"q={q:.2f} Tc={r['o1'][0]:.4f} x_o1={r['o1'][1]:.3f} "
              f"x_o2={r['o2'][1] if r['o2'] else float('nan'):.3f} x_P={r['e1'][1]:.3f} "
              f"x_eps={xe:.3f} nu={r['nu'] if r['nu'] else float('inf'):.2f}")
    rows.append(r)
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "..", "data", "tm_exponents_p1.json"), "w"), indent=1)
