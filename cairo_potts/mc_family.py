"""Parallel-tempering heat-bath MC for the interpolating family (request 2).

Identical algorithm to mc.py (A sites sampled, B summed exactly, exact
A-marginal weights in the replica exchange), but the plaquette type comes
from a kind map K[x, y] in {0 H, 1 V, 2 HU, 3 VU}.  In addition to the binned
observables, per-replica histograms of n3 and E (per site) are accumulated
for the first-order tests (bimodality and its size dependence).
"""
import numpy as np
from numba import njit

from .model import Q, costs, plaquette_tables, _plaquette_terms

KINDS = ("H", "V", "HU", "VU")
OBS = ["E", "E2", "Evar", "n3", "n32", "nb", "M", "M2", "M4", "absM"]


def tables(beta, p, q):
    out = {k: [] for k in ("logW", "e", "n3", "nb", "evar")}
    for kind in KINDS:
        t = plaquette_tables(beta, p, q, kind)
        en = np.array(_plaquette_terms(p, q, kind)[0])
        w = np.exp(-beta * (en - t["emin"][..., None, None]))
        Z = w.sum(axis=(4, 5))
        m1 = (w * en).sum(axis=(4, 5)) / Z
        m2 = (w * en ** 2).sum(axis=(4, 5)) / Z
        for k in ("logW", "e", "n3", "nb"):
            out[k].append(t[k])
        out["evar"].append(m2 - m1 ** 2)
    return {k: np.stack(v) for k, v in out.items()}


def kind_map(L, sel):
    return np.array([[((x + y) & 1) + (0 if sel(x, y) else 2) for y in range(L)]
                     for x in range(L)], np.int64)


@njit(cache=True)
def _local_logw(a, x, y, logW, mbE, L, K):
    xm = (x - 1) % L; ym = (y - 1) % L; xp = (x + 1) % L; yp = (y + 1) % L
    out = np.empty(Q)
    for c in range(Q):
        s = mbE[c]
        s += logW[K[x, y], c, a[xp, y], a[x, yp], a[xp, yp]]
        s += logW[K[xm, y], a[xm, y], c, a[xm, yp], a[x, yp]]
        s += logW[K[x, ym], a[x, ym], a[xp, ym], c, a[xp, y]]
        s += logW[K[xm, ym], a[xm, ym], a[x, ym], a[xm, y], c]
        out[c] = s
    return out


@njit(cache=True)
def sweep(a, logW, mbE, L, K, rnd):
    for x in range(L):
        for y in range(L):
            lw = _local_logw(a, x, y, logW, mbE, L, K)
            mx = max(lw[0], max(lw[1], lw[2]))
            w0 = np.exp(lw[0] - mx); w1 = np.exp(lw[1] - mx); w2 = np.exp(lw[2] - mx)
            r = rnd[x * L + y] * (w0 + w1 + w2)
            a[x, y] = 0 if r < w0 else (1 if r < w0 + w1 else 2)


@njit(cache=True)
def total_logw(a, logW, mbE, L, K):
    s = 0.0
    for x in range(L):
        for y in range(L):
            xp = (x + 1) % L; yp = (y + 1) % L
            s += mbE[a[x, y]] + logW[K[x, y], a[x, y], a[xp, y], a[x, yp], a[xp, yp]]
    return s


@njit(cache=True)
def measure(a, e, evar, n3t, nbt, Ea, L, K, tmpl, out):
    """tmpl[x, y]: order-parameter template (tiled ground-state pattern of
    delta(a,1) - delta(a,2)); M = sum phi * tmpl / sum tmpl^2."""
    E = 0.0; V = 0.0; n3 = 0.0; nb = 0.0; M = 0.0; norm = 0.0
    for x in range(L):
        for y in range(L):
            xp = (x + 1) % L; yp = (y + 1) % L
            k = K[x, y]
            c00 = a[x, y]; c10 = a[xp, y]; c01 = a[x, yp]; c11 = a[xp, yp]
            E += e[k, c00, c10, c01, c11] + Ea[c00]
            V += evar[k, c00, c10, c01, c11]
            n3 += n3t[k, c00, c10, c01, c11] + (c00 == 2)
            nb += nbt[k, c00, c10, c01, c11]
            t = tmpl[x, y]
            M += t * ((c00 == 0) * 1.0 - (c00 == 1) * 1.0)
            norm += t * t
    N = 3.0 * L * L
    out[0] = E / N; out[1] = (E / N) ** 2; out[2] = V / N ** 2
    out[3] = n3 / N; out[4] = (n3 / N) ** 2; out[5] = nb / N
    if norm > 0:
        M /= norm
    out[6] = M; out[7] = M * M; out[8] = M ** 4; out[9] = abs(M)


class FamilyPT:
    H_N3 = np.linspace(0.0, 0.4, 801)     # histogram edges, n3 per site
    H_E = np.linspace(0.0, 0.6, 1201)     # histogram edges, E per site

    def __init__(self, L, Ts, p, q, sel, seed=0, template=None):
        self.L, self.p, self.q = L, p, q
        if template is None:
            self.tmpl = np.zeros((L, L))
        else:
            px, py = template.shape
            if L % px or L % py:
                raise ValueError("L must be a multiple of the template period")
            self.tmpl = np.tile(template, (L // px, L // py)).astype(float)
        self.K = kind_map(L, sel)
        self.Ts = np.array(sorted(Ts)); self.R = len(self.Ts)
        self.rng = np.random.default_rng(seed)
        self.tab = [tables(1 / T, p, q) for T in self.Ts]
        self.Ea = costs(p, q)
        self.cfg = self.rng.integers(0, Q, size=(self.R, L, L)).astype(np.int64)
        self.where = np.arange(self.R)
        self.acc = np.zeros(self.R - 1); self.tries = np.zeros(self.R - 1)
        self.hn3 = np.zeros((self.R, len(self.H_N3) - 1))
        self.hE = np.zeros((self.R, len(self.H_E) - 1))

    def step(self):
        L, K = self.L, self.K
        for i in range(self.R):
            sweep(self.cfg[self.where[i]], self.tab[i]["logW"], -self.Ea / self.Ts[i], L, K,
                  self.rng.random(L * L))
        for i in range(self.rng.integers(2), self.R - 1, 2):
            ri, rj = self.where[i], self.where[i + 1]
            bi, bj = -self.Ea / self.Ts[i], -self.Ea / self.Ts[i + 1]
            Wi, Wj = self.tab[i]["logW"], self.tab[i + 1]["logW"]
            d = (total_logw(self.cfg[rj], Wi, bi, L, K) + total_logw(self.cfg[ri], Wj, bj, L, K)
                 - total_logw(self.cfg[ri], Wi, bi, L, K) - total_logw(self.cfg[rj], Wj, bj, L, K))
            self.tries[i] += 1
            if d >= 0 or self.rng.random() < np.exp(d):
                self.where[i], self.where[i + 1] = rj, ri
                self.acc[i] += 1

    def run(self, n_therm, n_bins, bin_size, callback=None):
        tmp = np.zeros((self.R, len(OBS)))
        for _ in range(n_therm):
            self.step()
        self.acc[:] = 0; self.tries[:] = 0
        bins = np.zeros((n_bins, self.R, len(OBS)))
        for b in range(n_bins):
            acc = np.zeros((self.R, len(OBS)))
            for _ in range(bin_size):
                self.step()
                for i in range(self.R):
                    t = self.tab[i]
                    measure(self.cfg[self.where[i]], t["e"], t["evar"], t["n3"], t["nb"],
                            self.Ea, self.L, self.K, self.tmpl, tmp[i])
                    self.hn3[i, min(max(np.searchsorted(self.H_N3, tmp[i, 3]) - 1, 0), len(self.H_N3) - 2)] += 1
                    self.hE[i, min(max(np.searchsorted(self.H_E, tmp[i, 0]) - 1, 0), len(self.H_E) - 2)] += 1
                acc += tmp
            bins[b] = acc / bin_size
            if callback is not None:
                callback(b, bins[: b + 1])
        return bins
