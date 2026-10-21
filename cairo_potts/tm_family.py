"""Transfer matrix (finite T) and min-plus transfer matrix (T = 0) for the
interpolating family: plaquette (x, y) is paired iff sel(x, y).

The transfer runs along x and the circumference L runs along y.  With column
stripes sel(x, y) = (x mod n != n - 1) the structure is periodic in x with
period P = lcm(2, n) columns and periodic in y with period 2, so any even L
is allowed for the cylinder.  sel(x, y) = True recovers the Cairo model of
tm.py (checked in scripts/family_kink.py).

Kind index of a plaquette: 0 = H, 1 = V, 2 = HU, 3 = VU (U = unpaired).
"""
from math import gcd

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from .model import Q, costs, plaquette_tables, _plaquette_terms
from .tm import _subspace_iteration

KINDS = ("H", "V", "HU", "VU")


def stripe_sel(n):
    """x = (n-1)/n: plaquettes in columns x = n-1 (mod n) are unpaired."""
    return (lambda x, y: (x % n) != n - 1), n


def column_sel(n, S):
    """General column layout: plaquettes in columns x with (x mod n) in S are
    paired (orientation still fixed by the parity of x + y).  x = |S| / n."""
    S = frozenset(S)
    return (lambda x, y: (x % n) in S), n


def parse_layout(spec):
    """'cairo' or 'n:s1,s2,...' -> (sel, n, x)."""
    if spec == "cairo":
        return (lambda x, y: True), 1, 1.0
    n, S = spec.split(":")
    n = int(n); S = [int(v) for v in S.split(",")]
    sel, n = column_sel(n, S)
    return sel, n, len(S) / n


def stripe_sel_low(n):
    """x = 1/n: only plaquettes in columns x = 0 (mod n) are paired."""
    return (lambda x, y: (x % n) == 0), n


def _corner(p, q):
    E = costs(p, q)
    return (E[:, None, None, None] + E[None, :, None, None]
            + E[None, None, :, None] + E[None, None, None, :]) / 4.0


def local_energy_tables(p, q):
    """T = 0 local plaquette energy (B pair minimised + E_a / 4 per corner)."""
    c = _corner(p, q)
    return [np.array(_plaquette_terms(p, q, k)[0]).min(axis=(4, 5)) + c for k in KINDS]


def _kind_index(x, y, sel):
    return ((x + y) & 1) + (0 if sel(x, y) else 2)


# ----------------------------------------------------------------- min-plus
def _minplus(X, lx, W, lw, out):
    """Y[out] = min over summed labels of X[lx] + W[lw]."""
    union = list(lx) + [l for l in lw if l not in lx]
    Xe = X.reshape(X.shape + (1,) * (len(union) - len(lx)))
    perm = [lw.index(l) for l in union if l in lw]
    Wt = np.transpose(W, perm)
    shape = [Q if l in lw else 1 for l in union]
    S = Xe + Wt.reshape(shape)
    red = tuple(i for i, l in enumerate(union) if l not in out)
    Y = S.min(axis=red)
    kept = [l for l in union if l in out]
    return np.transpose(Y, [kept.index(l) for l in out])


def minplus_column(v, x, L, em, sel):
    """v'(b) = min_a v(a) + E_x(a, b) for column x -> x+1."""
    X = v.reshape((Q,) * L)
    a = list(range(L)); b = list(range(L, 2 * L))
    labels = list(a)
    for y in range(L):
        yn = (y + 1) % L
        Wl = [a[y], b[y], a[yn], b[yn]]
        out = labels + [l for l in Wl if l not in labels]
        if y > 0:
            out.remove(a[y])
        if y == L - 1:
            out.remove(a[0])
        X = _minplus(X, labels, em[_kind_index(x, y, sel)], Wl, out)
        labels = out
    X = np.transpose(X, [labels.index(l) for l in b])
    return X.reshape(-1)


def ground_energy_family(L, p, q, sel, n, iters=200, return_vector=False):
    """Exact T = 0 energy per site on an L-cylinder (minimum cycle mean of the
    period-P min-plus transfer matrix)."""
    P = 2 * n // gcd(2, n)
    em = local_energy_tables(p, q)
    v = np.zeros(Q ** L)
    shifts = []
    for _ in range(iters):
        for x in range(P):
            v = minplus_column(v, x, L, em, sel)
        m = v.min(); shifts.append(m); v = v - m
    e = np.mean(shifts[iters // 2:]) / (3 * P * L)
    return (e, v) if return_vector else e


# ----------------------------------------------------------------- finite T
class FamilyTM:
    def __init__(self, L, beta, p, q, sel, n, e_shift=None):
        if L % 2:
            raise ValueError("L must be even")
        self.L, self.beta, self.p, self.q, self.sel, self.n = L, beta, p, q, sel, n
        self.P = 2 * n // gcd(2, n)
        c = _corner(p, q)
        tabs = [plaquette_tables(beta, p, q, k) for k in KINDS]
        logs = [t["logW"] - beta * c for t in tabs]
        emins = [t["emin"] + c for t in tabs]
        if e_shift is None:
            e_shift = max(min(e.min() for e in emins),
                          3 * ground_energy_family(4, p, q, sel, n, iters=60))
        self.e0 = e_shift
        self.W = [np.exp(lg + beta * self.e0) for lg in logs]

    def apply_column(self, v, x):
        L = self.L
        X = v.reshape((Q,) * L)
        a = list(range(L)); b = list(range(L, 2 * L))
        labels = list(a)
        for y in range(L):
            yn = (y + 1) % L
            Wl = [a[y], b[y], a[yn], b[yn]]
            out = labels + [l for l in Wl if l not in labels]
            if y > 0:
                out.remove(a[y])
            if y == L - 1:
                out.remove(a[0])
            X = np.einsum(X, labels, self.W[_kind_index(x, y, self.sel)], Wl, out, optimize=True)
            labels = out
        return np.transpose(X, [labels.index(l) for l in b]).reshape(-1)

    def shift_y(self, v, s):
        X = v.reshape((Q,) * self.L)
        return np.moveaxis(X, list(range(self.L)), [(y + s) % self.L for y in range(self.L)]).reshape(-1)

    def project_k0(self, v):
        acc = np.zeros_like(v)
        for s in range(0, self.L, 2):
            acc += self.shift_y(v, s)
        return acc / (self.L // 2)

    def apply_TP(self, v):
        for x in range(self.P):
            v = self.apply_column(v, x)
        return v

    def spectrum(self, k=4, tol=1e-12, k0=True):
        n = Q ** self.L
        proj = self.project_k0 if k0 else (lambda v: v)
        mv = lambda v: proj(self.apply_TP(v))
        v = proj(np.ones(n) + 0.01 * np.random.default_rng(1).random(n))
        for _ in range(3):
            w = mv(v); scale = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
        op = LinearOperator((n, n), matvec=lambda x: mv(x) / scale, dtype=float)
        try:
            lam = eigs(op, k=k, which="LM", tol=tol, v0=v, ncv=max(3 * k, 30),
                       return_eigenvectors=False)
        except Exception:
            lam, _ = _subspace_iteration(op.matvec, n, k, proj)
        lam = lam[np.argsort(-np.abs(lam))] * scale
        return lam

    def free_energy(self, k=2):
        lam = self.spectrum(k=k)
        f = -np.log(np.abs(lam[0])) / (3 * self.P * self.L * self.beta) + self.e0 / 3
        # scaled gaps per unit length along x (P columns per step)
        x = self.L * np.log(np.abs(lam[0]) / np.abs(lam[1:])) / (2 * np.pi * self.P)
        return f, x


def family_f(L, T, p, q, sel, n, e_shift=None):
    return FamilyTM(L, 1 / T, p, q, sel, n, e_shift).free_energy()[0]


def family_n3(L, T, p, q, sel, n, h=1e-4):
    # common shift for both evaluations so the difference is exact
    e = FamilyTM(L, 1 / T, p, q, sel, n).e0
    return (family_f(L, T, p, q + h, sel, n, e) - family_f(L, T, p, q - h, sel, n, e)) / (2 * h)


def column_energy_matrices(L, p, q, sel, n):
    """Dense T = 0 column transfer energies E_x(a, b) for x = 0..P-1 (L <= 6)."""
    P = 2 * n // gcd(2, n)
    em = local_energy_tables(p, q)
    idx = np.indices((Q,) * L).reshape(L, -1)
    Ms = []
    for x in range(P):
        M = np.zeros((Q ** L, Q ** L))
        for y in range(L):
            yn = (y + 1) % L
            M += em[_kind_index(x, y, sel)][idx[y][:, None], idx[y][None, :], idx[yn][:, None], idx[yn][None, :]]
        Ms.append(M)
    return Ms


def torus_ground_energy(Lx, L, p, q, sel, n):
    """Exact ground energy per site of the Lx x L torus (Lx multiple of P)."""
    Ms = column_energy_matrices(L, p, q, sel, n)
    P = len(Ms)
    def mp(A, B):
        return (A[:, :, None] + B[None, :, :]).min(axis=1)
    acc = Ms[0]
    for x in range(1, Lx):
        acc = mp(acc, Ms[x % P])
    return np.diag(acc).min() / (3 * Lx * L)


def torus_ground_states(Lx, L, p, q, sel, n, max_states=None):
    """Optimal A configurations on an Lx x L torus by min-plus dynamic
    programming with backtracking, one per optimal starting column state.
    Returns (E/N, list of arrays a[x, y])."""
    Ms = column_energy_matrices(L, p, q, sel, n)
    P = len(Ms)
    S = Q ** L
    idx = np.indices((Q,) * L).reshape(L, -1)
    # forward DP from every start state simultaneously
    cost = Ms[0].copy()                       # cost[s, b] after column 0
    back = []
    for x in range(1, Lx):
        tot = cost[:, :, None] + Ms[x % P][None, :, :]   # [s, a, b]
        back.append(tot.argmin(axis=1))
        cost = tot.min(axis=1)
    diag = np.diag(cost)
    best = diag.min()
    starts = np.where(np.isclose(diag, best))[0]
    if max_states:
        starts = starts[:max_states]
    confs = []
    for s in starts:
        cols = [s]
        b = s
        for x in range(Lx - 1, 0, -1):
            b = back[x - 1][s, b]
            cols.append(b)
        cols = cols[::-1]        # column states for x = 1..Lx (x = Lx is x = 0)
        cols = cols[-1:] + cols[:-1]   # reorder to x = 0..Lx-1
        confs.append(np.array([idx[:, c] for c in cols]))
    return best / (3 * Lx * L), confs


def colour3_sets(a, p, q, sel):
    """Set of colour-3 sites (A and B) of a configuration, B restored by
    minimising each plaquette (ties -> fewest colour-3 B)."""
    Lx, L = a.shape
    terms = {k: _plaquette_terms(p, q, k) for k in KINDS}
    S = {("A", x, y) for x in range(Lx) for y in range(L) if a[x, y] == 2}
    for x in range(Lx):
        for y in range(L):
            k = KINDS[_kind_index(x, y, sel)]
            cn = (a[x, y], a[(x + 1) % Lx, y], a[x, (y + 1) % L], a[(x + 1) % Lx, (y + 1) % L])
            en = np.array(terms[k][0])[cn]
            n3 = np.array(terms[k][1])[cn]
            opt = np.argwhere(np.isclose(en, en.min()))
            c, cp = min(opt, key=lambda t: n3[t[0], t[1]])
            if c == 2:
                S.add(("B", x, y, 0))
            if cp == 2:
                S.add(("B", x, y, 1))
    return S


def column_energy_count_matrices(L, p, q, sel, n, tol=1e-9):
    """Column transfer matrices in the (min, count) semiring: minimal energy
    E_x(a, b) and the number of optimal B assignments realising it."""
    P = 2 * n // gcd(2, n)
    c = _corner(p, q)
    full = [np.array(_plaquette_terms(p, q, k)[0]) for k in KINDS]
    em = [f.min(axis=(4, 5)) + c for f in full]
    mult = [np.isclose(f, f.min(axis=(4, 5), keepdims=True), atol=tol).sum(axis=(4, 5)) for f in full]
    idx = np.indices((Q,) * L).reshape(L, -1)
    out = []
    for x in range(P):
        E = np.zeros((Q ** L, Q ** L)); M = np.ones((Q ** L, Q ** L))
        for y in range(L):
            yn = (y + 1) % L
            k = _kind_index(x, y, sel)
            ix = (idx[y][:, None], idx[y][None, :], idx[yn][:, None], idx[yn][None, :])
            E += em[k][ix]; M *= mult[k][ix]
        out.append((E, M))
    return out


def torus_ground_count(Lx, L, p, q, sel, n, tol=1e-9):
    """(E0 per site, log number of ground states) on the Lx x L torus,
    counting A configurations and degenerate optimal B pairs."""
    mats = column_energy_count_matrices(L, p, q, sel, n, tol)
    P = len(mats)
    E, M = mats[0]
    logM = np.log(M)
    S = E.shape[0]
    chunk = max(1, int(3e7 // (S * S)))
    for x in range(1, Lx):
        E2, M2 = mats[x % P]
        lM2 = np.log(M2)
        En = np.empty_like(E); Ln = np.empty_like(logM)
        for s0 in range(0, S, chunk):     # chunk over start states (memory)
            sl = slice(s0, s0 + chunk)
            tot = E[sl, :, None] + E2[None, :, :]
            mn = tot.min(axis=1)
            lw = np.where(np.isclose(tot, mn[:, None, :], atol=tol),
                          logM[sl, :, None] + lM2[None, :, :], -np.inf)
            mx = lw.max(axis=1)
            Ln[sl] = mx + np.log(np.exp(lw - mx[:, None, :]).sum(axis=1))
            En[sl] = mn
        E, logM = En, Ln
    d = np.diag(E); best = d.min()
    on = np.isclose(d, best, atol=tol)
    lw = np.diag(logM)[on]
    return best / (3 * Lx * L), lw.max() + np.log(np.exp(lw - lw.max()).sum())
