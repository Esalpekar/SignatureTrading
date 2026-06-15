"""Expected signature computed two independent ways: Monte Carlo and analytic."""
from __future__ import annotations

import numpy as np

from . import signature


def monte_carlo(paths_embedded, level):
    """Coordinate-wise mean of per-path signatures, plus per-coordinate SE.

    ``paths_embedded`` has shape ``(n_paths, n_points, n_channels)``. Returns
    ``(mean, se)`` where ``se = std / sqrt(n_paths)`` (the Monte-Carlo standard
    error). Error decays like ``1/sqrt(n_paths)`` and is *not* reduced by a
    finer time grid.
    """
    paths_embedded = np.asarray(paths_embedded, dtype=float)
    sigs = np.stack(
        [signature.sig(p, level) for p in paths_embedded], axis=0
    )
    mean = sigs.mean(axis=0)
    se = sigs.std(axis=0, ddof=1) / np.sqrt(sigs.shape[0])
    return mean, se


def analytic_gaussian(generator, n_channels, level):
    """Analytic expected signature ``exp_(x)(T * xi)`` for a Gaussian driver.

    ``generator`` is the already-``T``-scaled generator ``(g1, g2)`` (i.e.
    ``T * xi``), passed straight to :func:`signature.tensor_exp`.
    """
    return signature.tensor_exp(generator, n_channels, level)
