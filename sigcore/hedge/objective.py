"""Assemble the hedging objective from covectors and the expected signature.

Strategy: position ``theta_t = <ell, S_{0,t}>`` (a covector against the running
signature, supported on the *adapted* channels time and lag). Trading P&L is
``<ell.a, S>`` with ``a`` the trade (lead) letter. Loss covector
``L = f - p0*empty - ell.a``; the objective is
``R(ell) = E[P(L)] = sum_k c_k <L^{shuffle k}, E[S]>``.

For ``P(x)=x^2`` this is the quadratic ``const - 2 b.ell + ell^T A ell`` with the
Gram matrix ``A`` (== the conditioning probe's ``A``) and ``b``. For the convex
quartic, the gradient (deg 3) and Hessian (deg 2) are exact shuffle contractions.
"""
from __future__ import annotations

import itertools

import numpy as np

from . import covector as cov
from .embedding import TIME, LAG, TRADE_LETTER, N_CHANNELS


def strategy_basis(depth):
    """Adapted words (over channels time=0, lag=2) up to ``depth``, incl. empty.

    These are the words the strategy covector ``ell`` is supported on; the trade
    letter is appended later to form the P&L covector.
    """
    letters = (TIME, LAG)
    words = [()]
    for L in range(1, depth + 1):
        words.extend(itertools.product(letters, repeat=L))
    return words


def loss_covector(payoff_cov, p0, ell_cov):
    """``L = payoff - p0*empty - append(ell, trade_letter)``."""
    return cov.add(payoff_cov, {(): -p0},
                   cov.scale(cov.append_letter(ell_cov, TRADE_LETTER), -1.0))


def required_signature_level(loss_cov, max_power):
    """Highest signature level the objective touches: ``maxlen(L) * max_power``."""
    return cov.max_word_length(loss_cov) * max_power


class TruncationError(ValueError):
    """Raised when E[S] is too shallow for the requested objective."""


def check_truncation(loss_cov, max_power, available_level):
    """Guard the truncation-degree coupling: never silently drop shuffle terms.

    The objective ``sum_k c_k <L^{shuffle k}, E[S]>`` reaches words of length
    ``maxlen(L) * max_power``; if ``E[S]`` is shallower the contraction would
    silently miss terms, biasing the objective and possibly breaking convexity.
    """
    need = required_signature_level(loss_cov, max_power)
    if need > available_level:
        raise TruncationError(
            f"objective needs E[S] to level {need} "
            f"(maxlen(L)={cov.max_word_length(loss_cov)} x degree {max_power}), "
            f"but only level {available_level} is available; "
            f"recompute E[S] deeper or reduce strategy depth/penalty degree.")


def objective_value(loss_cov, penalty_coeffs, expected_sig):
    """``R = sum_k c_k <L^{shuffle k}, E[S]>`` for ``penalty_coeffs = {k: c_k}``."""
    total = 0.0
    for k, ck in penalty_coeffs.items():
        if ck == 0.0:
            continue
        total += ck * cov.contract(cov.shuffle_power(loss_cov, k),
                                   expected_sig, N_CHANNELS)
    return total


# --- mean-variance system ----------------------------------------------------

def mean_variance_required_level(payoff_cov, depth):
    """E[S] level the mean-variance assembly needs.

    ``A`` reaches ``2(depth+1)`` (two appended trades); ``b`` reaches
    ``maxlen(f) + (depth+1)`` (payoff shuffled with one appended trade).
    """
    maxlen_f = cov.max_word_length(payoff_cov)
    return max(2 * (depth + 1), maxlen_f + depth + 1)


def mean_variance_system(payoff_cov, p0, basis, expected_sig):
    """Gram matrix ``A`` and vector ``b`` for ``R(ell) = const - 2 b.ell + ell^T A ell``.

    ``A_{wv} = <(w.a) shuffle (v.a), E[S]>``,
    ``b_w   = <(f - p0*empty) shuffle (w.a), E[S]>``.
    """
    depth = cov.max_word_length({w: 1.0 for w in basis})
    need = mean_variance_required_level(payoff_cov, depth)
    need_len = sum(N_CHANNELS ** k for k in range(1, need + 1))
    if len(expected_sig) < need_len:
        raise TruncationError(
            f"mean-variance system needs E[S] to level {need} (len {need_len}), "
            f"but the supplied E[S] has len {len(expected_sig)}; recompute deeper.")
    trades = [cov.append_letter({w: 1.0}, TRADE_LETTER) for w in basis]
    f0 = cov.add(payoff_cov, {(): -p0})
    n = len(basis)
    A = np.empty((n, n))
    b = np.empty(n)
    for i in range(n):
        b[i] = cov.contract(cov.shuffle_cov(f0, trades[i]), expected_sig, N_CHANNELS)
        for j in range(i, n):
            a_ij = cov.contract(cov.shuffle_cov(trades[i], trades[j]),
                                expected_sig, N_CHANNELS)
            A[i, j] = A[j, i] = a_ij
    return A, b


# --- general convex penalty: build loss, gradient, Hessian -------------------

def build_loss(ell_vec, basis, payoff_cov, p0):
    ell_cov = {w: ell_vec[i] for i, w in enumerate(basis)}
    return loss_covector(payoff_cov, p0, ell_cov)


def gradient(ell_vec, basis, payoff_cov, p0, penalty_coeffs, expected_sig):
    """``dR/dell_w = sum_k c_k k <L^{shuffle(k-1)} shuffle (-(w.a)), E[S]>``."""
    loss = build_loss(ell_vec, basis, payoff_cov, p0)
    trades = [cov.append_letter({w: 1.0}, TRADE_LETTER) for w in basis]
    g = np.zeros(len(basis))
    for k, ck in penalty_coeffs.items():
        if ck == 0.0 or k < 1:
            continue
        lk = cov.shuffle_power(loss, k - 1)
        for i, t in enumerate(trades):
            g[i] += ck * k * cov.contract(cov.shuffle_cov(lk, cov.scale(t, -1.0)),
                                          expected_sig, N_CHANNELS)
    return g


def hessian(ell_vec, basis, payoff_cov, p0, penalty_coeffs, expected_sig):
    """``d2R/dell_w dell_v = sum_k c_k k(k-1) <L^{shuffle(k-2)} shuffle (w.a) shuffle (v.a), E[S]>``."""
    loss = build_loss(ell_vec, basis, payoff_cov, p0)
    trades = [cov.append_letter({w: 1.0}, TRADE_LETTER) for w in basis]
    n = len(basis)
    H = np.zeros((n, n))
    for k, ck in penalty_coeffs.items():
        if ck == 0.0 or k < 2:
            continue
        lk = cov.shuffle_power(loss, k - 2)
        # precompute lk shuffled with each trade, then pair
        lk_t = [cov.shuffle_cov(lk, t) for t in trades]
        for i in range(n):
            for j in range(i, n):
                val = ck * k * (k - 1) * cov.contract(
                    cov.shuffle_cov(lk_t[i], trades[j]), expected_sig, N_CHANNELS)
                H[i, j] += val
                if j != i:
                    H[j, i] += val
    return H
