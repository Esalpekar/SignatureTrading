"""Level-wise signature scaling for conditioning.

The Gram matrix ``A`` is ill-conditioned (the conditioning probe shows ~100x per
level). Scaling is a diagonal change of basis on the strategy coefficients that
equalises coordinate magnitudes — here the empirical Jacobi scaling
``g_w = 1/sqrt(A_ww)`` — applied before the solve and undone afterwards. It is a
similarity-preserving rescale: ``A_s = G A G``, ``b_s = G b``, solve, then
``ell = G ell_s``. Applied consistently it leaves ``<ell, S>`` unchanged.
"""
from __future__ import annotations

import numpy as np


def scaling_factors(A):
    d = np.diag(A).astype(float).copy()
    d[d <= 0] = 1.0
    return 1.0 / np.sqrt(d)


def scale_system(A, b):
    """Return ``(A_s, b_s, g)`` with ``A_s = G A G``, ``b_s = G b``."""
    g = scaling_factors(A)
    A_s = A * g[:, None] * g[None, :]
    b_s = b * g
    return A_s, b_s, g


def ridged_solve(A, b, lam=0.0):
    """Solve ``(A + ...)ell = b`` in scaled coordinates with ridge ``lam``,
    then map back. The ridge acts on the *scaled* (equalised) coordinates."""
    A_s, b_s, g = scale_system(A, b)
    n = A_s.shape[0]
    ell_s = np.linalg.solve(A_s + lam * np.eye(n), b_s)
    return g * ell_s


def condition_numbers(A):
    """``(cond(A), cond(A_s))`` — unscaled vs scaled, for the H4 report."""
    A_s, _, _ = scale_system(A, np.zeros(A.shape[0]))
    return float(np.linalg.cond(A)), float(np.linalg.cond(A_s))
