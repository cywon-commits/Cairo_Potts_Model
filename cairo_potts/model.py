"""Plaquette representation of the Cairo Potts model.

A sites sit on the integer square lattice (x, y).  Every edge of that lattice
carries one B site.  Plaquette (x, y), with corners

    a00 = a(x, y)    a10 = a(x+1, y)
    a01 = a(x, y+1)  a11 = a(x+1, y+1)

pairs the B sites of two opposite edges and bonds them to each other:

    x + y even  ->  'H' : bottom edge (a00-a10) with top edge (a01-a11)
    x + y odd   ->  'V' : left edge  (a00-a01) with right edge (a10-a11)

Every B belongs to exactly one plaquette, so summing out B gives an exact
plaquette weight W(a00, a10, a01, a11).  Arrays below are indexed
[a00, a10, a01, a11].

Interpolating family: an *unpaired* plaquette keeps the same two B sites
(same orientation rule) but drops their B-B bond, so its weight factorises
into two 2-site factors.  Kinds "HU" / "VU" are the unpaired versions.
"""
import numpy as np

Q = 3


def costs(p, q):
    return np.array([0.0, p, q])


def _plaquette_terms(p, q, kind):
    """Return arrays over (a00,a10,a01,a11,c,c') of
    energy, number of colour-3 B sites, for the B pair (c, c')."""
    E = costs(p, q)
    a = np.arange(Q)
    d = (a[:, None] == a[None, :]).astype(float)  # delta(i,j)
    # broadcasting axes: a00, a10, a01, a11, c, cp
    # integer arrays: bool + bool would be a logical OR
    A00 = a.reshape(Q, 1, 1, 1, 1, 1)
    A10 = a.reshape(1, Q, 1, 1, 1, 1)
    A01 = a.reshape(1, 1, Q, 1, 1, 1)
    A11 = a.reshape(1, 1, 1, Q, 1, 1)
    C = a.reshape(1, 1, 1, 1, Q, 1)
    CP = a.reshape(1, 1, 1, 1, 1, Q)
    paired = not kind.endswith("U")
    orient = kind[0]
    if orient == "H":   # c on (a00,a10), c' on (a01,a11)
        bonds = (C == A00) * 1 + (C == A10) + (CP == A01) + (CP == A11)
    elif orient == "V":  # c on (a00,a01), c' on (a10,a11)
        bonds = (C == A00) * 1 + (C == A01) + (CP == A10) + (CP == A11)
    else:
        raise ValueError(kind)
    if paired:
        bonds = bonds + (C == CP)
    en = bonds + E[C] + E[CP]
    n3 = (C == 2).astype(float) + (CP == 2)
    n2 = (C == 1).astype(float) + (CP == 1)
    nb = bonds.astype(float)
    shape = (Q,) * 6
    return (np.broadcast_to(en, shape), np.broadcast_to(n3, shape),
            np.broadcast_to(n2, shape), np.broadcast_to(nb, shape))


def plaquette_tables(beta, p, q, kind):
    """Exact B-integrated plaquette data for one plaquette type.

    Returns dict with
      W   : weight sum_{c,c'} exp(-beta e)                [a00,a10,a01,a11]
      logW: log of W, computed stably
      e   : <B-pair energy | A corners>
      n3  : <# colour-3 among the two B | A corners>
      n2  : <# colour-2 among the two B | A corners>
      nb  : <# unsatisfied bonds in plaquette | A corners>
      emin: zero-temperature plaquette energy min_{c,c'} e
    """
    en, n3, n2, nb = _plaquette_terms(p, q, kind)
    emin = en.min(axis=(4, 5))
    w = np.exp(-beta * (en - emin[..., None, None]))
    Z = w.sum(axis=(4, 5))
    out = dict(
        logW=np.log(Z) - beta * emin,
        e=(w * en).sum(axis=(4, 5)) / Z,
        n3=(w * n3).sum(axis=(4, 5)) / Z,
        n2=(w * n2).sum(axis=(4, 5)) / Z,
        nb=(w * nb).sum(axis=(4, 5)) / Z,
        emin=emin,
    )
    out["W"] = np.exp(out["logW"])
    return out
