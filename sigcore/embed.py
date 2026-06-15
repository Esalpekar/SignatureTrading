"""Path embeddings: time augmentation and the Hoff lead-lag transform."""
from __future__ import annotations

import numpy as np


def time_augment(times, series):
    """Stack ``times`` and ``series`` into a 2-channel path.

    ``series`` may be a single series shape ``(n_steps+1,)`` or a batch
    ``(n_paths, n_steps+1)``. Returns shape ``(..., n_steps+1, 2)`` with
    channel 0 = time, channel 1 = value.
    """
    series = np.asarray(series, dtype=float)
    times = np.asarray(times, dtype=float)

    if series.ndim == 1:
        return np.stack([times, series], axis=-1)

    # Batched: broadcast times across paths.
    t = np.broadcast_to(times, series.shape)
    return np.stack([t, series], axis=-1)


def lead_lag(series):
    """Hoff lead-lag transform of a scalar series.

    Produces a 2-channel ``(lead, lag)`` path traced as a staircase: at each
    original sample we first advance the lead channel, then the lag channel,
    so the lead is held one sample ahead of the lag. Correctness is defined by
    test T2: the antisymmetric part of the level-2 signature equals
    ``1/2 * sum_k (x_{k+1} - x_k)^2`` (half the realised quadratic variation).

    NOTE: not wired into the pricing path; validated here only to de-risk the
    hedging phase.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.size

    lead = np.empty(2 * n - 1)
    lag = np.empty(2 * n - 1)

    # Staircase: from node 2k, step lead up to x[k+1] (node 2k+1),
    # then step lag up to x[k+1] (node 2k+2).
    lead[0] = x[0]
    lag[0] = x[0]
    for k in range(n - 1):
        lead[2 * k + 1] = x[k + 1]
        lag[2 * k + 1] = x[k]
        lead[2 * k + 2] = x[k + 1]
        lag[2 * k + 2] = x[k + 1]

    return np.stack([lead, lag], axis=-1)
