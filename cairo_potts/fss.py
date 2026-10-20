"""Phenomenological renormalisation and conformal-invariance analysis built on
the symmetric column transfer matrix."""
import numpy as np
from scipy.optimize import brentq

from .tm import SymmetricTM


def gaps(L, T, p, q, k=6):
    """Scaled gaps x_i = L ln|mu0/mu_i| / 2pi, signs of mu_i, and the reduced
    free energy per unit area (per A site) F = -ln mu0_true / L."""
    t = SymmetricTM(L, 1.0 / T, p, q)
    mu = t.spectrum_G(k=k)
    x = L * np.log(np.abs(mu[0]) / np.abs(mu[1:])) / (2 * np.pi)
    F = -(np.log(np.abs(mu[0])) - t.beta * L * t.e0) / L
    return dict(x=x, sign=np.sign(mu.real[1:]), F=F, mu=mu)


def crossing(L1, L2, p, q, Tlo, Thi, level=0, k=6, xtol=1e-5):
    """Temperature where x_level(L1) = x_level(L2) (phenomenological RG)."""
    f = lambda T: gaps(L1, T, p, q, k)["x"][level] - gaps(L2, T, p, q, k)["x"][level]
    Tc = brentq(f, Tlo, Thi, xtol=xtol)
    g1, g2 = gaps(L1, Tc, p, q, k), gaps(L2, Tc, p, q, k)
    return Tc, g1, g2


def central_charge(Ls, T, p, q):
    """Fit F(L) = F_inf - pi c / (6 L^2) + b / L^4 to the three largest L."""
    F = np.array([gaps(L, T, p, q, k=2)["F"] for L in Ls])
    Ls = np.asarray(Ls, float)
    A = np.stack([np.ones_like(Ls), -np.pi / (6 * Ls ** 2), Ls ** -4.0], 1)
    sol = np.linalg.lstsq(A[-3:], F[-3:], rcond=None)[0]
    # two-point estimates for successive pairs (no 1/L^4 term)
    c2 = [(F[i] - F[i + 1]) / (np.pi / 6 * (1 / Ls[i + 1] ** 2 - 1 / Ls[i] ** 2))
          for i in range(len(Ls) - 1)]
    return sol[1], c2, F
