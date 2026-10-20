"""Exact transfer matrix of the B-integrated plaquette model on a cylinder.

Transfer runs along x; the circumference L (even) runs along y with periodic
boundary conditions.  The plaquette type alternates with x + y, so the
elementary period is two columns:  T2 = T(odd column) . T(even column).

The matrix is never stored for L > 6: the matrix-vector product is applied
plaquette by plaquette (one 3^(L+2) intermediate) and fed to ARPACK.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from .model import Q, costs, plaquette_tables


_E0_CACHE = {}


def ground_energy(p, q, L=4, iters=300):
    """Exact T = 0 energy per site on period-compatible states, from the
    min-plus transfer matrix on an L = 4 cylinder (all known ground states
    have period 2)."""
    key = (round(p, 12), round(q, 12), L)
    if key in _E0_CACHE:
        return _E0_CACHE[key]
    from .model import _plaquette_terms
    E = costs(p, q)
    corner = (E[:, None, None, None] + E[None, :, None, None]
              + E[None, None, :, None] + E[None, None, None, :]) / 4.0
    em = {k: np.array(_plaquette_terms(p, q, k)[0]).min(axis=(4, 5)) + corner
          for k in ("H", "V")}
    n = Q ** L
    idx = np.indices((Q,) * L).reshape(L, -1)          # column states
    def column_energy(xpar):
        M = np.zeros((n, n))
        for y in range(L):
            k = "H" if (xpar + y) % 2 == 0 else "V"
            a0, a1 = idx[y], idx[(y + 1) % L]
            M += em[k][a0[:, None], idx[y][None, :], a1[:, None], idx[(y + 1) % L][None, :]]
        return M
    Me, Mo = column_energy(0), column_energy(1)
    v = np.zeros(n)
    shifts = []
    for _ in range(iters):
        v = (v[:, None] + Me).min(0)
        v = (v[:, None] + Mo).min(0)
        shifts.append(v.min())
        v -= v.min()
    # minimum cycle mean per T2 step (any transient period averages out)
    e = np.mean(shifts[iters // 2:]) / (6 * L)
    _E0_CACHE[key] = e
    return e


def _subspace_iteration(matvec, n, k, project, m=None, iters=400, tol=1e-12):
    m = m or 2 * k + 2
    rng = np.random.default_rng(7)
    V = np.linalg.qr(np.stack([project(rng.random(n)) for _ in range(m)], 1))[0]
    prev = None
    for it in range(iters):
        W = np.stack([matvec(V[:, j]) for j in range(m)], 1)
        H = V.T @ W
        V, _ = np.linalg.qr(W)
        if it % 10 == 9:
            ev = np.linalg.eigvals(H)
            ev = ev[np.argsort(-np.abs(ev))][:k]
            if prev is not None and np.allclose(ev, prev, rtol=tol, atol=tol * abs(ev[0])):
                break
            prev = ev
    W = np.stack([matvec(V[:, j]) for j in range(m)], 1)
    H = V.T @ W
    ev, S = np.linalg.eig(H)
    o = np.argsort(-np.abs(ev))[:k]
    return ev[o], V @ S[:, o]


class CylinderTM:
    def __init__(self, L, beta, p, q, x_kind=None):
        if L % 2:
            raise ValueError("circumference must be even (checkerboard pairing)")
        self.L, self.beta, self.p, self.q = L, beta, p, q
        tabH = plaquette_tables(beta, p, q, "H")
        tabV = plaquette_tables(beta, p, q, "V")
        # Each A site is a corner of exactly four plaquettes: hand E_a / 4 to
        # each of them, so that a plaquette carries its full local energy.
        # Shifting by the smallest local plaquette energy then keeps the
        # weights of (near-)ground-state columns O(1) even at very low T.
        E = costs(p, q)
        corner = (E[:, None, None, None] + E[None, :, None, None]
                  + E[None, None, :, None] + E[None, None, None, :]) / 4.0
        logH = tabH["logW"] - beta * corner
        logV = tabV["logW"] - beta * corner
        emH = tabH["emin"] + corner
        emV = tabV["emin"] + corner
        # shift by the exact T = 0 energy per plaquette (3 E0) when it is
        # larger than the smallest local energy (frustrated decompositions)
        self.e0 = max(min(emH.min(), emV.min()), 3 * ground_energy(p, q))
        self.WH = np.exp(logH + beta * self.e0)
        self.WV = np.exp(logV + beta * self.e0)
        self.site = np.ones(Q)

    # plaquette (x, y) is H when x + y is even
    def _kind(self, x, y):
        return self.WH if (x + y) % 2 == 0 else self.WV

    def apply_column(self, v, x):
        """v'(b) = sum_a v(a) T_x(a, b): column x (labels a) -> column x+1 (b)."""
        L = self.L
        X = v.reshape((Q,) * L)
        a = list(range(L))             # old-column labels
        b = list(range(L, 2 * L))      # new-column labels
        labels = list(a)
        for y in range(L):
            yn = (y + 1) % L
            Wl = [a[y], b[y], a[yn], b[yn]]
            keep_new = [l for l in Wl if l not in labels]
            out = labels + keep_new
            # drop the old label a[y] once no later plaquette needs it
            if y > 0:
                out.remove(a[y])
            if y == L - 1:
                out.remove(a[0])
            X = np.einsum(X, labels, self._kind(x, y), Wl, out, optimize=True)
            labels = out
        # reorder to b_0 .. b_{L-1} and apply on-site weights of new column
        perm = [labels.index(l) for l in b]
        X = np.transpose(X, perm)
        for y in range(L):
            shape = [1] * L
            shape[y] = Q
            X = X * self.site.reshape(shape)
        return X.reshape(-1)

    def apply_T2(self, v):
        return self.apply_column(self.apply_column(v, 0), 1)

    def dense_T2(self):
        n = Q ** self.L
        M = np.empty((n, n))
        I = np.eye(n)
        for i in range(n):
            M[i] = self.apply_T2(I[i])
        return M  # row i = e_i^T T2

    def spectrum(self, k=4, dense_max=6, tol=1e-13):
        """Largest-modulus eigenvalues of T2, sorted by modulus (descending)."""
        n = Q ** self.L
        if self.L <= dense_max:
            ev = np.linalg.eigvals(self.dense_T2())
        else:
            op = LinearOperator((n, n), matvec=self.apply_T2, dtype=float)
            v0 = np.ones(n)
            ev = eigs(op, k=k, which="LM", tol=tol, v0=v0,
                      ncv=max(2 * k + 1, 20), return_eigenvectors=False)
        ev = ev[np.argsort(-np.abs(ev))]
        return ev[:k]

    def observables(self, k=4, **kw):
        """Free energy per site f (N = 3 N_A) and correlation length xi in
        units of the A-lattice spacing along the cylinder."""
        ev = self.spectrum(k=k, **kw)
        lam0, lam1 = np.abs(ev[0]), np.abs(ev[1])
        f = -np.log(lam0) / (6 * self.L * self.beta) + self.e0 / 3.0
        xi = 2.0 / np.log(lam0 / lam1)
        return dict(f=f, xi=xi, ev=ev)


def free_energy(L, T, p, q, **kw):
    return CylinderTM(L, 1.0 / T, p, q).observables(**kw)["f"]


def n3_density(L, T, p, q, h=1e-4, **kw):
    """Colour-3 density per site, n3 = df/dq (central difference)."""
    return (free_energy(L, T, p, q + h, **kw) - free_energy(L, T, p, q - h, **kw)) / (2 * h)


def n2_density(L, T, p, q, h=1e-4, **kw):
    return (free_energy(L, T, p + h, q, **kw) - free_energy(L, T, p - h, q, **kw)) / (2 * h)


class SymmetricTM(CylinderTM):
    """Single-column transfer G = S . T(column 0), S = shift y -> y+1.

    G^2 = T2 . S^2 and S^2 (translation by two rows) commutes with G, so in
    the zero-momentum sector of S^2 the eigenvalues of G are square roots of
    those of T2 and the sign of a real eigenvalue is the parity under the
    lattice symmetry 'translate by (1,1)'.  The Z4 clock magnetisation of the
    Potts phase is odd under it, the polarisation (expensive colour on one A
    checkerboard) is even.
    """

    def shift_y(self, v, s=1):
        X = v.reshape((Q,) * self.L)
        return np.roll(X, s, axis=tuple(range(self.L))[0:1] and None) if False else \
            np.moveaxis(X, list(range(self.L)), [(y + s) % self.L for y in range(self.L)]).reshape(-1)

    def project_k0(self, v):
        acc = np.zeros_like(v)
        for s in range(0, self.L, 2):
            acc += self.shift_y(v, s)
        return acc / (self.L // 2)

    def apply_G(self, v):
        return self.project_k0(self.shift_y(self.apply_column(v, 0), 1))

    def apply_T2k0(self, v):
        return self.project_k0(self.apply_T2(v))

    def spectrum_G(self, k=8, tol=1e-12, return_vectors=False):
        """Leading eigenvalues mu of G (|mu| descending).

        G itself is strongly non-normal on the ordered-vacuum subspace (it
        swaps vacua that live at very different scales in the column basis),
        which breaks Arnoldi at low T.  We therefore diagonalise the k = 0
        block of T2 = G^2, which is well conditioned, and recover the sign of
        mu = +-sqrt(lambda) from the Rayleigh quotient of G on the eigenvector.
        Complex or ambiguous cases are returned as +sqrt(lambda) * phase.
        """
        n = Q ** self.L
        v = self.project_k0(np.ones(n) + 0.01 * np.random.default_rng(1).random(n))
        for _ in range(4):  # normalise the operator so that |lambda_0| ~ 1
            w = self.apply_T2k0(v)
            scale = np.linalg.norm(w) / np.linalg.norm(v)
            v = w / np.linalg.norm(w)
        op = LinearOperator((n, n), matvec=lambda x: self.apply_T2k0(x) / scale,
                            dtype=float)
        try:
            lam, vec = eigs(op, k=k, which="LM", tol=tol, v0=v, ncv=max(3 * k, 30))
        except Exception:
            # Krylov breakdown at very low T (operator numerically low rank):
            # fall back to orthogonal subspace iteration + Rayleigh-Ritz.
            lam, vec = _subspace_iteration(op.matvec, n, k, self.project_k0)
        order = np.argsort(-np.abs(lam))
        lam, vec = lam[order], vec[:, order]
        mu = np.sqrt(lam.astype(complex))
        for i in range(len(lam)):
            x = vec[:, i]
            gx = self.apply_G(x.real) + 1j * self.apply_G(x.imag)
            r = np.vdot(x, gx) / np.vdot(x, x)
            if abs(r) > 0.5 * abs(mu[i]) * np.sqrt(scale) and np.real(r) < 0:
                mu[i] = -mu[i]
        mu = mu * np.sqrt(scale)
        return mu
