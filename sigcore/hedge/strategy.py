"""Evaluate the feedback strategy ``theta_t = <ell, S_{0,t}>`` along a path.

Uses iisignature's stream signature (one call gives every prefix) and reads the
position at the *adapted* staircase node ``2k`` for each price index ``k``.
"""
from __future__ import annotations

import numpy as np
import iisignature

from sigcore import signature
from . import embedding as emb
from .embedding import N_CHANNELS


def _contract_sig(ell_cov, sig_vec):
    total = ell_cov.get((), 0.0)
    for w, v in ell_cov.items():
        if w:
            total += v * sig_vec[signature.word_index(w, N_CHANNELS)]
    return total


def positions(ell_cov, times, price, level):
    """Positions ``theta_k = <ell, S_{0,t_k}>`` for each price index ``k``.

    ``price`` is a single series ``(n,)`` on grid ``times``; returns ``(n,)``.
    ``theta_0`` uses the empty signature (constant part of ``ell``).
    """
    price = np.asarray(price, float)
    n = price.size
    enlarged = emb.enlarged(times, price)
    theta = np.empty(n)
    theta[0] = ell_cov.get((), 0.0)
    if n == 1:
        return theta
    stream = iisignature.sig(enlarged, level, 2)     # rows: prefixes of len 2..m
    for k in range(1, n):
        # adapted node 2k -> prefix of 2k+1 points -> stream row (2k+1)-2
        theta[k] = _contract_sig(ell_cov, stream[2 * k - 1])
    return theta
