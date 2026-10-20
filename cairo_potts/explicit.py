"""The original Cairo model with B sites kept explicitly (verification #3).

The graph is built directly: A sites on an Lx x Ly torus, one B site on every
square-lattice edge, each B bonded to the two A ends of its edge, and the two
B sites of opposite edges of every plaquette bonded to each other
(x + y even: bottom-top, odd: left-right).
"""
import numpy as np
from numba import njit

from .model import Q, costs


def cairo_graph(Lx, Ly):
    A = lambda x, y: (x % Lx) * Ly + (y % Ly)
    NA = Lx * Ly
    h = lambda x, y: NA + 2 * A(x, y)        # B on edge (x,y)-(x+1,y)
    v = lambda x, y: NA + 2 * A(x, y) + 1    # B on edge (x,y)-(x,y+1)
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            bonds += [(h(x, y), A(x, y)), (h(x, y), A(x + 1, y)),
                      (v(x, y), A(x, y)), (v(x, y), A(x, y + 1))]
            if (x + y) % 2 == 0:
                bonds.append((h(x, y), h(x, y + 1)))
            else:
                bonds.append((v(x, y), v(x + 1, y)))
    N = 3 * NA
    is_A = np.zeros(N, bool); is_A[:NA] = True
    return N, np.array(bonds), is_A


def neighbour_table(N, bonds):
    nb = [[] for _ in range(N)]
    for i, j in bonds:
        nb[i].append(j); nb[j].append(i)
    deg = np.array([len(n) for n in nb])
    tab = -np.ones((N, deg.max()), np.int64)
    for i, n in enumerate(nb):
        tab[i, :len(n)] = n
    return tab, deg


def exact_enumeration(Lx, Ly, T, p, q):
    """Brute-force <E>/N, <n3>, ln Z on a tiny torus (N <= 12)."""
    N, bonds, _ = cairo_graph(Lx, Ly)
    E_site = costs(p, q)
    confs = np.indices((Q,) * N).reshape(N, -1).T
    E = (confs[:, bonds[:, 0]] == confs[:, bonds[:, 1]]).sum(1) + E_site[confs].sum(1)
    w = np.exp(-(E - E.min()) / T)
    Z = w.sum()
    return dict(E=(w * E).sum() / Z / N, n3=(w * (confs == 2).sum(1)).sum() / Z / N,
                lnZ=np.log(Z) - E.min() / T)


@njit(cache=True)
def _sweep(s, nbt, deg, Esite, beta, rnd):
    N = s.shape[0]
    for i in range(N):
        w = np.empty(3)
        for c in range(3):
            e = Esite[c]
            for k in range(deg[i]):
                if s[nbt[i, k]] == c:
                    e += 1.0
            w[c] = e
        m = min(w[0], min(w[1], w[2]))
        w0 = np.exp(-beta * (w[0] - m)); w1 = np.exp(-beta * (w[1] - m)); w2 = np.exp(-beta * (w[2] - m))
        r = rnd[i] * (w0 + w1 + w2)
        s[i] = 0 if r < w0 else (1 if r < w0 + w1 else 2)


@njit(cache=True)
def _energy(s, bonds, Esite):
    e = 0.0
    for b in range(bonds.shape[0]):
        if s[bonds[b, 0]] == s[bonds[b, 1]]:
            e += 1.0
    for i in range(s.shape[0]):
        e += Esite[s[i]]
    return e


def explicit_mc(Lx, Ly, T, p, q, n_sweeps, n_therm=2000, seed=0, n_bins=50):
    """Single-site heat bath with explicit B.  Returns bin averages of
    E/N and n3."""
    N, bonds, _ = cairo_graph(Lx, Ly)
    nbt, deg = neighbour_table(N, bonds)
    Es = costs(p, q)
    rng = np.random.default_rng(seed)
    s = rng.integers(0, Q, N).astype(np.int64)
    for _ in range(n_therm):
        _sweep(s, nbt, deg, Es, 1 / T, rng.random(N))
    per = n_sweeps // n_bins
    E = np.zeros(n_bins); n3 = np.zeros(n_bins)
    for b in range(n_bins):
        for _ in range(per):
            _sweep(s, nbt, deg, Es, 1 / T, rng.random(N))
            E[b] += _energy(s, bonds, Es) / N
            n3[b] += (s == 2).sum() / N
    return E / per, n3 / per
