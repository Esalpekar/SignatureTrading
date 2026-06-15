"""Multi-asset signature hedging (2-3 assets).

Enlarged path: time-augmented + Hoff lead-lag of EACH price channel, so
``d = 1 + 2M`` channels: ``time``; and per asset ``m``: ``lead_m``, ``lag_m``.
Trade letter ``a_m`` = ``lead_m`` (appending it gives the Ito integral
``int theta^m dX^m`` — validated by M-I). Each asset has its own strategy
covector ``ell^m`` against the JOINT running signature (cross-asset feedback);
the mean-variance dependency matrix is block-structured by asset pair, and the
off-diagonal blocks are the entire cross-asset story.

Reuses the channel-agnostic covector algebra; everything is parameterised by the
channel count ``d`` and the per-asset trade letters.
"""
from __future__ import annotations

import itertools

import numpy as np
import iisignature

from sigcore import signature
from . import covector as cov

TIME = 0


def layout(M):
    leads = [1 + 2 * m for m in range(M)]
    lags = [2 + 2 * m for m in range(M)]
    return {"M": M, "d": 1 + 2 * M, "leads": leads, "lags": lags}


# --- enlarged path -----------------------------------------------------------

def enlarged_batch(times, paths_list):
    """Joint enlarged path, shape ``(n_paths, 2n-1, 1+2M)``.

    ``paths_list`` is a list of ``M`` arrays shape ``(n_paths, n)`` on grid
    ``times``. Time advances on lag (even) steps; each asset's lead leads its lag.
    """
    t = np.asarray(times, float)
    M = len(paths_list)
    p, n = paths_list[0].shape
    m = 2 * n - 1
    cols = []
    tt = np.empty((p, m))
    tt[:, 0] = t[0]
    tt[:, 1::2] = t[:-1]
    tt[:, 2::2] = t[1:]
    cols.append(tt)
    for X in paths_list:
        X = np.asarray(X, float)
        lead = np.empty((p, m))
        lag = np.empty((p, m))
        lead[:, 0] = X[:, 0]
        lag[:, 0] = X[:, 0]
        lead[:, 1::2] = X[:, 1:]
        lag[:, 1::2] = X[:, :-1]
        lead[:, 2::2] = X[:, 1:]
        lag[:, 2::2] = X[:, 1:]
        cols.append(lead)
        cols.append(lag)
    return np.stack(cols, axis=-1)


def enlarged(times, series_list):
    batched = enlarged_batch(times, [np.asarray(s, float)[None, :] for s in series_list])
    return batched[0]


def expected_signature(times, paths_list, level):
    emb = enlarged_batch(times, paths_list)
    sigs = iisignature.sig(np.ascontiguousarray(emb), level)
    return sigs.mean(axis=0), sigs.std(axis=0, ddof=1) / np.sqrt(sigs.shape[0])


# --- payoff covectors on the joint path --------------------------------------

def basket_forward_covector(x0, weights, K, M):
    """``sum_i w_i X^i_T - K``: ``(sum w_i x0_i - K)*empty + sum_i w_i*(lead_i,)``."""
    lay = layout(M)
    const = float(np.dot(weights, x0) - K)
    out = {(): const}
    for i, w in enumerate(weights):
        if w != 0.0:
            out[(lay["leads"][i],)] = float(w)
    return out


def spread_forward_covector(x0, K, i, j, M):
    """``X^i_T - X^j_T - K``."""
    w = np.zeros(M)
    w[i], w[j] = 1.0, -1.0
    return basket_forward_covector(x0, w, K, M)


def basket_asian_covector(x0, weights, K, T, M):
    """``sum_i w_i A^i - K``, ``A^i = (1/T) int X^i dt`` (the (lag_i, time) word)."""
    lay = layout(M)
    const = float(np.dot(weights, x0) - K)
    out = {(): const}
    for i, w in enumerate(weights):
        if w != 0.0:
            out[(lay["lags"][i], TIME)] = float(w) / T
    return out


# --- strategy basis & block mean-variance ------------------------------------

def strategy_basis(M, depth):
    """Adapted words for each ell^m: over {time} U {lag_1..lag_M} (cross feedback)."""
    lay = layout(M)
    letters = [TIME] + lay["lags"]
    words = [()]
    for L in range(1, depth + 1):
        words.extend(itertools.product(letters, repeat=L))
    return words


def _trade_cov(word, lead_letter):
    return cov.append_letter({word: 1.0}, lead_letter)


def mean_variance_system(payoff_cov, p0, basis, M, expected_sig):
    """Block-structured ``A`` and ``b`` over the stacked index ``(asset, word)``.

    Returns ``(A, b, index)`` where ``index[(m, w)] = row``. Blocks:
    ``A_{(m,w),(n,v)} = <(w.a_m) shuffle (v.a_n), E[S]>``,
    ``b_{(m,w)} = <(f - p0*empty) shuffle (w.a_m), E[S]>``.
    """
    lay = layout(M)
    d = lay["d"]
    f0 = cov.add(payoff_cov, {(): -p0})
    stacked = [(m, w) for m in range(M) for w in basis]
    trades = {(m, w): _trade_cov(w, lay["leads"][m]) for (m, w) in stacked}
    n = len(stacked)
    A = np.empty((n, n))
    b = np.empty(n)
    for i, key_i in enumerate(stacked):
        b[i] = cov.contract(cov.shuffle_cov(f0, trades[key_i]), expected_sig, d)
        for j in range(i, n):
            a_ij = cov.contract(cov.shuffle_cov(trades[key_i], trades[stacked[j]]),
                                expected_sig, d)
            A[i, j] = A[j, i] = a_ij
    index = {key: i for i, key in enumerate(stacked)}
    return A, b, index


def loss_covector(payoff_cov, p0, ell_by_asset, M):
    """``L = f - p0*empty - sum_m append(ell^m, a_m)``.

    ``ell_by_asset`` is a list of ``M`` covector dicts.
    """
    lay = layout(M)
    terms = [payoff_cov, {(): -p0}]
    for m in range(M):
        terms.append(cov.scale(cov.append_letter(ell_by_asset[m], lay["leads"][m]), -1.0))
    return cov.add(*terms)


def objective_value(loss_cov, penalty_coeffs, expected_sig, M):
    d = layout(M)["d"]
    total = 0.0
    for k, ck in penalty_coeffs.items():
        if ck:
            total += ck * cov.contract(cov.shuffle_power(loss_cov, k), expected_sig, d)
    return total


# --- general convex penalty: stacked gradient / Hessian ----------------------

def _stacked_to_assets(vec, basis, M):
    nb = len(basis)
    return [dict(zip(basis, vec[m * nb:(m + 1) * nb])) for m in range(M)]


def gradient(vec, basis, payoff_cov, p0, penalty_coeffs, expected_sig, M):
    lay = layout(M)
    d = lay["d"]
    ell = _stacked_to_assets(vec, basis, M)
    loss = loss_covector(payoff_cov, p0, ell, M)
    stacked = [(m, w) for m in range(M) for w in basis]
    trades = [_trade_cov(w, lay["leads"][m]) for (m, w) in stacked]
    g = np.zeros(len(stacked))
    for k, ck in penalty_coeffs.items():
        if not ck or k < 1:
            continue
        lk = cov.shuffle_power(loss, k - 1)
        for i, tr in enumerate(trades):
            g[i] += ck * k * cov.contract(cov.shuffle_cov(lk, cov.scale(tr, -1.0)),
                                          expected_sig, d)
    return g


def hessian(vec, basis, payoff_cov, p0, penalty_coeffs, expected_sig, M):
    lay = layout(M)
    d = lay["d"]
    ell = _stacked_to_assets(vec, basis, M)
    loss = loss_covector(payoff_cov, p0, ell, M)
    stacked = [(m, w) for m in range(M) for w in basis]
    trades = [_trade_cov(w, lay["leads"][m]) for (m, w) in stacked]
    n = len(stacked)
    H = np.zeros((n, n))
    for k, ck in penalty_coeffs.items():
        if not ck or k < 2:
            continue
        lk = cov.shuffle_power(loss, k - 2)
        lk_t = [cov.shuffle_cov(lk, tr) for tr in trades]
        for i in range(n):
            for j in range(i, n):
                val = ck * k * (k - 1) * cov.contract(
                    cov.shuffle_cov(lk_t[i], trades[j]), expected_sig, d)
                H[i, j] += val
                if j != i:
                    H[j, i] += val
    return H


# --- strategy evaluation & P&L -----------------------------------------------

def positions(ell_by_asset, times, price_list, level, M):
    """Per-asset positions ``theta^m_k = <ell^m, S_{0,t_k}>`` along one path.

    ``price_list`` is a list of ``M`` series (each ``(n,)``). Returns array
    ``(M, n)``. theta at price index 0 uses the empty signature.
    """
    lay = layout(M)
    d = lay["d"]
    n = len(times)
    enl = enlarged(times, price_list)
    theta = np.zeros((M, n))
    for m in range(M):
        theta[m, 0] = ell_by_asset[m].get((), 0.0)
    if n == 1:
        return theta
    stream = iisignature.sig(enl, level, 2)
    for k in range(1, n):
        row = stream[2 * k - 1]
        for m in range(M):
            tot = ell_by_asset[m].get((), 0.0)
            for w, v in ell_by_asset[m].items():
                if w:
                    tot += v * row[signature.word_index(w, d)]
            theta[m, k] = tot
    return theta


def shortfall(ell_by_asset, payoff_fn, p0, times, paths_list, level, M,
              rebalance_steps=None):
    """Per-path shortfall ``L = payoff - p0 - sum_m sum_k theta^m_k dX^m_k``.

    P&L uses literal simulated increments; payoff via ``payoff_fn(times, paths_list)``.
    """
    payoff = payoff_fn(times, paths_list)
    n_paths = paths_list[0].shape[0]
    if rebalance_steps is None or rebalance_steps >= len(times) - 1:
        rt = times
        rp = paths_list
    else:
        idx = np.unique(np.linspace(0, len(times) - 1, rebalance_steps + 1).round().astype(int))
        rt = times[idx]
        rp = [X[:, idx] for X in paths_list]
    pnl = np.zeros(n_paths)
    for p in range(n_paths):
        series = [X[p] for X in rp]
        theta = positions(ell_by_asset, rt, series, max(level, 1), M)
        for m in range(M):
            pnl[p] += np.sum(theta[m, :-1] * np.diff(series[m]))
    return payoff - p0 - pnl
