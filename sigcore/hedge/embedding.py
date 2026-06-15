"""The enlarged path: time-augmented Hoff lead-lag of price.

Channels: ``0 = time`` (advances on lag steps), ``1 = lead``, ``2 = lag``.
The Hoff staircase makes ``lead`` move one sub-step ahead of ``lag``; appending
the lead letter to a strategy covector therefore yields the *Ito* (left-point)
trading integral — the mechanism validated by core test I.

Verified coordinate facts (see the hedging tests):
  <(1,), S>   = X_T - X_0        (terminal price increment)
  <(0,), S>   = T                (time)
  <(2,1), S>  = int (X-X0) dX    (Ito; integrand = lag, integrate d(lead))
  <(2,0), S>  = int (X-X0) dt    (trapezoidal running average -> Asian)
At price index k the adapted info is the prefix up to staircase node ``2k``.
"""
from __future__ import annotations

import numpy as np
import iisignature

TIME, LEAD, LAG = 0, 1, 2
TRADE_LETTER = LEAD
N_CHANNELS = 3


def enlarged(times, series):
    """Single enlarged path, shape ``(2n-1, 3)`` for an ``n``-point price series."""
    return enlarged_batch(times, np.asarray(series, float)[None, :])[0]


def enlarged_batch(times, paths):
    """Batched enlarged paths, shape ``(n_paths, 2n-1, 3)``.

    ``paths`` is ``(n_paths, n)``; ``times`` is ``(n,)``.
    """
    paths = np.asarray(paths, float)
    t = np.asarray(times, float)
    p, n = paths.shape
    m = 2 * n - 1

    lead = np.empty((p, m))
    lag = np.empty((p, m))
    tt = np.empty((p, m))

    lead[:, 0] = paths[:, 0]
    lag[:, 0] = paths[:, 0]
    tt[:, 0] = t[0]
    # odd nodes (2k+1): lead has jumped to X_{k+1}, lag still X_k, time = t_k
    lead[:, 1::2] = paths[:, 1:]
    lag[:, 1::2] = paths[:, :-1]
    tt[:, 1::2] = t[:-1]
    # even nodes (2k+2): lag catches up to X_{k+1}, time advances to t_{k+1}
    lead[:, 2::2] = paths[:, 1:]
    lag[:, 2::2] = paths[:, 1:]
    tt[:, 2::2] = t[1:]

    return np.stack([tt, lead, lag], axis=-1)


def adapted_node(k):
    """Staircase node index whose prefix carries the info available at price k."""
    return 2 * k


def expected_signature(times, paths, level):
    """Monte-Carlo expected signature of the enlarged path and per-coord SE."""
    emb = enlarged_batch(times, paths)
    sigs = iisignature.sig(np.ascontiguousarray(emb), level)
    return sigs.mean(axis=0), sigs.std(axis=0, ddof=1) / np.sqrt(sigs.shape[0])


# --- payoff covectors on the enlarged path (closed form from the core) -------
# Represented as dicts {word(tuple): coeff}; the empty word () is the constant.

def forward_covector(x0, K):
    """Forward paying ``X_T - K``: ``(x0-K)*empty + 1*(price increment)``."""
    return {(): x0 - K, (LEAD,): 1.0}


def asian_covector(x0, K, T):
    """Arithmetic-average forward paying ``A - K``, ``A = (1/T) int X dt``.

    ``A - X0 = (1/T) <(lag, time), S>`` (the trapezoidal running average).
    """
    return {(): x0 - K, (LAG, TIME): 1.0 / T}
