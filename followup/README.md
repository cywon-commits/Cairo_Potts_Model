# Follow-up: geometry of the three-colour region and Ising criticality at its bound

This folder accompanies the follow-up to Paper I (Cairo lattice). Paper I closed the
phase diagram of one lattice. This work asks which shapes the zero-temperature
three-colour region of the antiferromagnetic three-state Potts model with on-site
costs (D2, D3) can take on any lattice, and what happens at the edge of that range.

Notation: J = 1. In the fundamental sector, p' is the largest cost relative to the
cheapest one and q' is the middle cost. The mirror line is q' = p'.

## Main results and where they are checked

| Claim | Status | Script(s) | Output |
|---|---|---|---|
| Three-colour region R = union of −∂B (convex duality); each edge is one competing move | proof (standard coexistence argument) | `code/fam/analyze.py`, `code/fam/hexcheck2.py` | `figures/fig1_gallery.png`, `figures/fig2_duality.png` |
| Theorem 1: the axis threshold equals the marginal unheal cost | proof | `code/proof/hetero.py`, `code/proof/hetero_curve.py` | `figures/fig3_theorems.png` (a) |
| Theorem 2: single-site thresholds are integers ≥ 1 | proof | — | — |
| Theorem 3: the unit hexagon lies inside R against two-colour competitors; s ≥ 1 − b_min/f_exp | conditional proof; no counterexample in 1384 exhaustive graphs (n ≤ 10) and 1600 CP-SAT graphs (n = 11–20; planar, G(n,p), 3- and 4-regular) | `code/proof/twocol.py`, `code/proof/cpsat_check.py` | `data/logs/cps_*.log` |
| Mixed competitors (improper three-colour states) can enter the unit hexagon on general graphs | counterexamples (9 vertices exhaustive; n = 10–18 CP-SAT, including planar graphs, down to p' = 0.75) | `code/proof/mirror.py`, `code/proof/cpsat_full.py`, `code/proof/cpsat_kind.py` | `figures/fig3_theorems.png` (c), `data/logs/full*_*.log`, `data/logs/kind_*.log` |
| Truncated diced lattice: half phase in the wedge 1 < p' < 1 + q' | exact transfer matrix | `code/proof/halfphase.py`, `code/proof/half_grid.py` | `figures/fig3_theorems.png` (b) |
| Size and shape knobs: hub family, Cairo + bridge, floret, decorated diced (s down to −3/2, star shape) | hand prediction + exact transfer matrix | `code/fam/*`, `code/floret/*`, `code/ddice/ddice2.py` | `figures/hexagon_design_family.png`, `figures/shear_control.png`, `figures/continuous_shear.png`, `figures/concave_cc3.png`, `figures/concave_pure_C2.png` |
| Finite T: a corner with d > 1 is first order with t* = d − T Δs0/Δn; at d = 1 it is a crossover whose midpoint shift ln x* = 0.409 is fixed by an effective dice Ising model | low-T expansion + exact sum-product transfer matrix | `code/ddice/finiteT.py`, `code/ddice/fig6data.py`, `code/eff/eff.py` | `figures/fig6_corners.png` |
| Parallel units m ≥ 2: Ising line t_c(T) = 1 + T ln x_c(m), with x_c = (2 − r)/(r − 1), r = exp(2K_c/m) and K_c(dice) = 0.415721 | exact decimation → effective dice Ising (error ≈ 0.26 T e^{−1/T} per site) + χ/W^{7/4} crossings | `code/mpar/decim.py`, `code/mpar/ising_map.py`, `code/mpar/corr.py`, `code/mpar/chi2.py` | `figures/fig7_ising.png`, `code/mpar/corr.json`, `data/logs/chi_m*_T*.log` |

## Finite-size crossings versus the closed form

The table lists χ/W^{7/4} crossings of widths W = 3 and 4 against the predicted t_c.

| m | T | predicted t_c | crossing W = 3, 4 | (crossing − t_c)/T |
|---|---|---|---|---|
| 2 | 0.05 | 0.9969 | 0.9934 | −0.07 |
| 2 | 0.10 | 0.9938 | 0.9863 | −0.075 |
| 2 | 0.15 | 0.9907 | 0.9779 | −0.085 |
| 2 | 0.20 | 0.9876 | 0.9629 | −0.12 |
| 3 | 0.05 | 1.0378 | 1.0354 | −0.048 |
| 3 | 0.10 | 1.0757 | 1.0711 | −0.046 |
| 3 | 0.15 | 1.1135 | 1.1053 | −0.055 |
| 3 | 0.20 | 1.1513 | 1.1357 | −0.078 |
| 4 | 0.05 | 1.0601 | 1.0577 | −0.048 |
| 4 | 0.10 | 1.1202 | 1.1165 | −0.037 |

For T ≤ 0.1 the offset in ln x = (t − 1)/T does not depend on T, so it is a
finite-width effect of the effective Ising model itself. At T = 0.15–0.2 it grows,
because corrections of order T e^{−1/T} start to matter. The closed form therefore
holds quantitatively for T ≲ 0.15 J.

At low t some widths show a second, spurious crossing. It comes from strip-length
saturation of χ, so the scripts report the crossing nearest the prediction.

## Running

Each script adds its own folder to `sys.path` through `ROOT`, the `code/` directory.
Most can be run from any working directory. `mpar/corr.py` and `mpar/effchi.py` read
`ising_map.py` by relative path, so run those two from inside `code/mpar/`.

```bash
pip install numpy scipy networkx matplotlib ortools
python3 code/proof/cpsat_check.py 1 200      # Theorem 3 test, seed 1, 200 graphs
python3 code/mpar/chi2.py 3 0.1              # FSS crossing, m = 3, T = 0.1
cd code/figs && python3 fig7.py              # regenerate a figure
```

The transfer matrices eliminate sites one at a time on cylinders. Bulk quantities are
read from the far end of long strips, as differences of cumulative marks over whole
periods. This avoids the boundary transients that appear near first-order
coexistence.
