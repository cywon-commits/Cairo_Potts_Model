"""Exhaustive T = 0 search on small tori of the plaquette model, with the B
sites restored by minimising each plaquette."""
import itertools
import numpy as np
from numba import njit

from .model import Q, costs, _plaquette_terms


def plaquette_min_tables(p, q):
    """emin[kind][a00,a10,a01,a11] and the minimising B pair (c, c')."""
    out = {}
    for kind in ("H", "V"):
        en = np.array(_plaquette_terms(p, q, kind)[0])
        flat = en.reshape(Q ** 4, Q * Q)
        arg = flat.argmin(axis=1)
        nmin = (np.isclose(flat, flat.min(axis=1, keepdims=True))).sum(axis=1)
        out[kind] = (flat.min(axis=1).reshape((Q,) * 4),
                     np.stack([arg // Q, arg % Q], -1).reshape((Q,) * 4 + (2,)),
                     nmin.reshape((Q,) * 4))
    return out


@njit(cache=True)
def _enumerate(Lx, Ly, eH, eV, Ea, tol):
    n = Lx * Ly
    total = Q ** n
    best = 1e300
    a = np.zeros(n, np.int64)
    res = [np.int64(0)]
    for idx in range(total):
        t = idx
        for k in range(n):
            a[k] = t % Q
            t //= Q
        e = 0.0
        for x in range(Lx):
            for y in range(Ly):
                xp = (x + 1) % Lx
                yp = (y + 1) % Ly
                a00 = a[x * Ly + y]; a10 = a[xp * Ly + y]
                a01 = a[x * Ly + yp]; a11 = a[xp * Ly + yp]
                if (x + y) % 2 == 0:
                    e += eH[a00, a10, a01, a11]
                else:
                    e += eV[a00, a10, a01, a11]
                e += Ea[a[x * Ly + y]]
        if e < best - tol:
            best = e
            res = [np.int64(idx)]
        elif e < best + tol:
            res.append(np.int64(idx))
    return best, res


def ground_states(Lx, Ly, p, q):
    tabs = plaquette_min_tables(p, q)
    best, idxs = _enumerate(Lx, Ly, tabs["H"][0], tabs["V"][0], costs(p, q), 1e-9)
    confs = []
    for idx in idxs:
        a = np.array([(idx // Q ** k) % Q for k in range(Lx * Ly)]).reshape(Lx, Ly)
        confs.append(a)
    return best / (3 * Lx * Ly), confs, tabs


def restore_B(a, tabs):
    """Return dict plaquette (x,y) -> (kind, c, c', degeneracy)."""
    Lx, Ly = a.shape
    out = {}
    for x in range(Lx):
        for y in range(Ly):
            xp, yp = (x + 1) % Lx, (y + 1) % Ly
            k = "H" if (x + y) % 2 == 0 else "V"
            key = (a[x, y], a[xp, y], a[x, yp], a[xp, yp])
            c, cp = tabs[k][1][key]
            out[(x, y)] = (k, int(c), int(cp), int(tabs[k][2][key]))
    return out


@njit(cache=True)
def _enumerate_kinds(Lx, Ly, eK, kind, Ea, tol):
    """As _enumerate, with an explicit kind index per plaquette
    (0 H, 1 V, 2 HU, 3 VU); returns the minimum and all minimisers."""
    n = Lx * Ly
    total = Q ** n
    best = 1e300
    a = np.zeros(n, np.int64)
    res = [np.int64(0)]
    for idx in range(total):
        t = idx
        for k in range(n):
            a[k] = t % Q
            t //= Q
        e = 0.0
        for x in range(Lx):
            for y in range(Ly):
                xp = (x + 1) % Lx
                yp = (y + 1) % Ly
                e += eK[kind[x, y], a[x * Ly + y], a[xp * Ly + y], a[x * Ly + yp], a[xp * Ly + yp]]
                e += Ea[a[x * Ly + y]]
        if e < best - tol:
            best = e
            res = [np.int64(idx)]
        elif e < best + tol:
            res.append(np.int64(idx))
    return best, res


def ground_states_family(Lx, Ly, p, q, sel):
    """Exhaustive T = 0 search on an Lx x Ly torus for the family; the B
    pair of every plaquette is minimised, and n3 of each ground state counts
    the colour-3 A sites plus, per plaquette, the minimal and maximal number
    of colour-3 B sites over the degenerate optimal B pairs."""
    from .model import _plaquette_terms
    kinds = ("H", "V", "HU", "VU")
    terms = [_plaquette_terms(p, q, k) for k in kinds]
    eK = np.stack([np.array(t[0]).min(axis=(4, 5)) for t in terms])
    kind = np.array([[((x + y) & 1) + (0 if sel(x, y) else 2) for y in range(Ly)] for x in range(Lx)])
    best, idxs = _enumerate_kinds(Lx, Ly, eK, kind, costs(p, q), 1e-9)
    confs = []
    for idx in idxs:
        a = np.array([(idx // Q ** k) % Q for k in range(Lx * Ly)]).reshape(Lx, Ly)
        n3min = (a == 2).sum(); n3max = n3min
        for x in range(Lx):
            for y in range(Ly):
                k = kind[x, y]
                cn = (a[x, y], a[(x + 1) % Lx, y], a[x, (y + 1) % Ly], a[(x + 1) % Lx, (y + 1) % Ly])
                en = np.array(terms[k][0])[cn]
                b3 = np.array(terms[k][1])[cn]
                opt = np.isclose(en, en.min())
                n3min += b3[opt].min(); n3max += b3[opt].max()
        confs.append((a, n3min, n3max))
    return best / (3 * Lx * Ly), confs
