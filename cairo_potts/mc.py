"""Parallel-tempering heat-bath Monte Carlo of the B-integrated plaquette model
on an L x L torus of A sites (N = 3 L^2 sites in total).

* Only A sites are sampled; every plaquette's B pair is summed exactly, so
  the heat-bath probabilities and all B observables are exact conditional
  averages (improved estimators).
* Replica exchange uses the exact A-marginal weight
      w_beta(a) = exp(-beta sum_i E_{a_i}) prod_plaq W_beta(corners),
  which is not of the form exp(-beta E(a)); the swap test evaluates both
  configurations at both temperatures.
* Specific heat via the law of total variance:
      Var(E) = <Var(E | A)> + Var(<E | A>).

Order parameters (colour 1 = cheapest, code colour 0):
    sigma = (4/N_A) sum_A delta(a,1) (-1)^x      (colour-1 column density wave)
    tau   = (4/N_A) sum_A delta(a,1) (-1)^y      (colour-1 row density wave)
The four crystal states have |sigma| = |tau| = 1; the colour-1 line states
of the mirror line p = q have (|sigma|, |tau|) = (1, 0) or (0, 1).
"""
import numpy as np
from numba import njit

from .model import Q, costs, plaquette_tables

# per-bin observables, in this order
OBS = ["E", "E2", "Evar", "n3", "n2", "m2", "m4", "absm", "P", "absP", "P2", "P4",
       "Nem", "absNem", "Nem2", "Nem4", "c4m4", "sig2", "tau2", "nb"]


def tables(beta, p, q):
    """Stack H/V tables -> arrays indexed [kind, a00, a10, a01, a11]."""
    tH = plaquette_tables(beta, p, q, "H")
    tV = plaquette_tables(beta, p, q, "V")
    st = lambda k: np.stack([tH[k], tV[k]])
    en_var = []
    # conditional variance of the plaquette energy
    from .model import _plaquette_terms
    for kind, t in (("H", tH), ("V", tV)):
        en = np.array(_plaquette_terms(p, q, kind)[0])
        w = np.exp(-beta * (en - t["emin"][..., None, None]))
        Z = w.sum(axis=(4, 5))
        m1 = (w * en).sum(axis=(4, 5)) / Z
        m2 = (w * en ** 2).sum(axis=(4, 5)) / Z
        en_var.append(m2 - m1 ** 2)
    return dict(logW=st("logW"), e=st("e"), n3=st("n3"), n2=st("n2"), nb=st("nb"),
                evar=np.stack(en_var))


@njit(cache=True)
def _kind(x, y):
    return (x + y) & 1  # 0 = H, 1 = V


@njit(cache=True)
def _local_logw(a, x, y, logW, mbE, L):
    """log weight of the 3 colours at A site (x, y), other sites fixed."""
    xm = (x - 1) % L; ym = (y - 1) % L; xp = (x + 1) % L; yp = (y + 1) % L
    out = np.empty(Q)
    for c in range(Q):
        s = mbE[c]
        # plaquette (x, y): c is a00
        s += logW[_kind(x, y), c, a[xp, y], a[x, yp], a[xp, yp]]
        # plaquette (x-1, y): c is a10
        s += logW[_kind(xm, y), a[xm, y], c, a[xm, yp], a[x, yp]]
        # plaquette (x, y-1): c is a01
        s += logW[_kind(x, ym), a[x, ym], a[xp, ym], c, a[xp, y]]
        # plaquette (x-1, y-1): c is a11
        s += logW[_kind(xm, ym), a[xm, ym], a[x, ym], a[xm, y], c]
        out[c] = s
    return out


@njit(cache=True)
def sweep(a, logW, mbE, L, rnd):
    """One heat-bath sweep (random-free sequential order)."""
    for x in range(L):
        for y in range(L):
            lw = _local_logw(a, x, y, logW, mbE, L)
            mx = max(lw[0], max(lw[1], lw[2]))
            w0 = np.exp(lw[0] - mx); w1 = np.exp(lw[1] - mx); w2 = np.exp(lw[2] - mx)
            r = rnd[x * L + y] * (w0 + w1 + w2)
            if r < w0:
                a[x, y] = 0
            elif r < w0 + w1:
                a[x, y] = 1
            else:
                a[x, y] = 2


@njit(cache=True)
def total_logw(a, logW, mbE, L):
    s = 0.0
    for x in range(L):
        for y in range(L):
            xp = (x + 1) % L; yp = (y + 1) % L
            s += mbE[a[x, y]]
            s += logW[_kind(x, y), a[x, y], a[xp, y], a[x, yp], a[xp, yp]]
    return s


@njit(cache=True)
def measure(a, e, evar, n3t, n2t, nbt, Ea, L, out):
    E = 0.0; V = 0.0; n3 = 0.0; n2 = 0.0; nb = 0.0
    sig = 0.0; tau = 0.0
    for x in range(L):
        for y in range(L):
            xp = (x + 1) % L; yp = (y + 1) % L
            k = _kind(x, y)
            c00 = a[x, y]; c10 = a[xp, y]; c01 = a[x, yp]; c11 = a[xp, yp]
            E += e[k, c00, c10, c01, c11] + Ea[c00]
            V += evar[k, c00, c10, c01, c11]
            n3 += n3t[k, c00, c10, c01, c11] + (c00 == 2)
            n2 += n2t[k, c00, c10, c01, c11] + (c00 == 1)
            nb += nbt[k, c00, c10, c01, c11]
            if c00 == 0:
                sig += 1.0 - 2.0 * (x & 1)
                tau += 1.0 - 2.0 * (y & 1)
    NA = L * L
    N = 3.0 * NA
    sig *= 4.0 / NA; tau *= 4.0 / NA
    m2 = sig * sig + tau * tau
    P = sig * tau
    Nem = sig * sig - tau * tau
    out[0] = E / N
    out[1] = (E / N) ** 2
    out[2] = V / N ** 2
    out[3] = n3 / N
    out[4] = n2 / N
    out[5] = m2
    out[6] = m2 * m2
    out[7] = np.sqrt(m2)
    out[8] = P
    out[9] = abs(P)
    out[10] = P * P
    out[11] = P ** 4
    out[12] = Nem
    out[13] = abs(Nem)
    out[14] = Nem * Nem
    out[15] = Nem ** 4
    out[16] = sig ** 4 - 6 * sig * sig * tau * tau + tau ** 4  # m^4 cos(4 theta)
    out[17] = sig * sig
    out[18] = tau * tau
    out[19] = nb / N


class PTRun:
    def __init__(self, L, Ts, p, q, seed=0, init="random"):
        if L % 2:
            raise ValueError("L must be even")
        self.L, self.p, self.q = L, p, q
        self.Ts = np.array(sorted(Ts))
        self.R = len(self.Ts)
        self.rng = np.random.default_rng(seed)
        self.tab = [tables(1 / T, p, q) for T in self.Ts]
        self.Ea = costs(p, q)
        self.cfg = self.rng.integers(0, Q, size=(self.R, L, L)).astype(np.int64)
        self.where = np.arange(self.R)          # replica index holding temperature i
        self.acc = np.zeros(self.R - 1); self.tries = np.zeros(self.R - 1)
        self.trips = np.zeros(self.R)           # round trips per replica (diagnostic)
        self.label = np.zeros(self.R, int)      # +1 last visited bottom, -1 top
        self.label[self.where[0]] = 1; self.label[self.where[-1]] = -1

    def _mbE(self, i):
        return -self.Ea / self.Ts[i]

    def step(self):
        L = self.L
        for i in range(self.R):
            r = self.where[i]
            sweep(self.cfg[r], self.tab[i]["logW"], self._mbE(i), L, self.rng.random(L * L))
        # replica exchange, alternating even / odd pairs
        start = self.rng.integers(2)
        for i in range(start, self.R - 1, 2):
            ri, rj = self.where[i], self.where[i + 1]
            lii = total_logw(self.cfg[ri], self.tab[i]["logW"], self._mbE(i), L)
            ljj = total_logw(self.cfg[rj], self.tab[i + 1]["logW"], self._mbE(i + 1), L)
            lij = total_logw(self.cfg[rj], self.tab[i]["logW"], self._mbE(i), L)
            lji = total_logw(self.cfg[ri], self.tab[i + 1]["logW"], self._mbE(i + 1), L)
            self.tries[i] += 1
            d = lij + lji - lii - ljj
            if d >= 0 or self.rng.random() < np.exp(d):
                self.where[i], self.where[i + 1] = rj, ri
                self.acc[i] += 1
        bot, top = self.where[0], self.where[-1]
        if self.label[bot] == -1:
            self.trips[bot] += 1
        self.label[bot] = 1; self.label[top] = -1

    def measure_all(self, out):
        for i in range(self.R):
            t = self.tab[i]
            measure(self.cfg[self.where[i]], t["e"], t["evar"], t["n3"], t["n2"], t["nb"],
                    self.Ea, self.L, out[i])

    def run(self, n_therm, n_bins, bin_size, callback=None):
        tmp = np.zeros((self.R, len(OBS)))
        for _ in range(n_therm):
            self.step()
        self.acc[:] = 0; self.tries[:] = 0; self.trips[:] = 0
        bins = np.zeros((n_bins, self.R, len(OBS)))
        for b in range(n_bins):
            acc = np.zeros((self.R, len(OBS)))
            for _ in range(bin_size):
                self.step()
                self.measure_all(tmp)
                acc += tmp
            bins[b] = acc / bin_size
            if callback is not None:
                callback(b, bins[: b + 1])
        return bins
