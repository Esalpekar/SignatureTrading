"""Shared test helpers: closed-form oracles, covectors, tensor algebra over
flat signatures, and path generators.

Kept out of ``sigcore`` originals so the v0 modules stay untouched; these are
test-side oracles and small primitives the suite needs.
"""
from __future__ import annotations

import itertools
from collections import Counter
from math import comb, factorial

import numpy as np
import iisignature

from sigcore import signature, embed, shuffle as _shuffle

R_EPS = 1e-8  # below this |r|, use the removable-singularity limit in the Asian


# --- embeddings --------------------------------------------------------------

def log_price_embed(times, paths):
    """Time-augmented log-price path(s): channel 0 = time, channel 1 = log X."""
    return embed.time_augment(times, np.log(paths))


def time_augment_multi(times, *series):
    """Stack time and several batched series into a multi-channel path.

    ``series`` are each shape ``(n_paths, n_points)``; result is
    ``(n_paths, n_points, 1 + len(series))`` with channel 0 = time.
    """
    t = np.broadcast_to(times, series[0].shape)
    return np.stack([t, *series], axis=-1)


def correlated_bm_oracle(rho, T, level):
    """Analytic E[S] of ``(t, W1, W2)`` with corr(W1, W2) = rho, at all levels.

    ``xi = e0 + 1/2 (e1(x)e1 + e2(x)e2 + rho(e1(x)e2 + e2(x)e1))`` (unit-variance
    drivers). Evaluated via ``signature.tensor_exp`` over 3 channels.
    """
    return correlated_oracle(np.array([[1.0, rho], [rho, 1.0]]), T, level)


def correlated_oracle(corr, T, level, sigmas=None, r=0.0):
    """Analytic E[S] of the time-augmented correlated path ``(t, Y^1..Y^M)``.

    Unit Brownian (``sigmas=None``): ``xi = e_time + 1/2 sum_ij corr_ij e_i(x)e_j``.
    Log-GBM (``sigmas`` given): ``xi = e_time + sum_i (r - sigma_i^2/2) e_i
    + 1/2 sum_ij sigma_i sigma_j corr_ij e_i(x)e_j``. The path has ``d = 1+M``
    channels (channel 0 = time); the cross coordinate satisfies
    ``E[<(i,j),S> + <(j,i),S>] = sigma_i sigma_j corr_ij T`` (= corr_ij T for BM).
    """
    corr = np.asarray(corr, dtype=float)
    M = corr.shape[0]
    d = 1 + M
    s = np.ones(M) if sigmas is None else np.asarray(sigmas, dtype=float)
    g1 = np.zeros(d)
    g1[0] = T
    if sigmas is not None:
        for i in range(M):
            g1[1 + i] = (r - 0.5 * s[i] ** 2) * T
    g2 = np.zeros((d, d))
    for i in range(M):
        for j in range(M):
            g2[1 + i, 1 + j] = 0.5 * s[i] * s[j] * corr[i, j] * T
    return signature.tensor_exp((g1, g2), d, level)


# --- Monte-Carlo expected signature (batched) --------------------------------

def mc_expected_signature(paths_embedded, level):
    """Mean per-path signature and per-coordinate standard error.

    Same quantity as ``expected_signature.monte_carlo`` but uses iisignature's
    batched path input so the heavy B/D tests (up to 1e5 paths x level 4) run
    in seconds. ``paths_embedded`` has shape ``(n_paths, n_points, n_channels)``.
    """
    paths_embedded = np.ascontiguousarray(paths_embedded, dtype=float)
    sigs = iisignature.sig(paths_embedded, level)        # (n_paths, sig_len)
    mean = sigs.mean(axis=0)
    se = sigs.std(axis=0, ddof=1) / np.sqrt(sigs.shape[0])
    return mean, se


# --- closed-form oracles -----------------------------------------------------

def gbm_log_expected_signature(r, sigma, T, level):
    """Analytic E[S] of the time-augmented log-price GBM ``(t, log X_t)``.

    ``log X`` is ABM with drift: ``xi = e0 + (r - sigma^2/2) e1 + (sigma^2/2) e1(x)e1``.
    Evaluated at all levels via the existing ``signature.tensor_exp``.
    """
    g1 = np.array([T, (r - 0.5 * sigma * sigma) * T])
    g2 = np.zeros((2, 2))
    g2[1, 1] = 0.5 * sigma * sigma * T
    return signature.tensor_exp((g1, g2), 2, level)


def asian_expected_average(x0, r, T):
    """E[A] for the arithmetic average ``A = (1/T) int_0^T X_t dt`` under GBM.

    ``E[A] = X0 (e^{rT} - 1)/(rT)`` with the removable singularity ``-> X0`` at
    ``r = 0`` handled explicitly.
    """
    if abs(r) < R_EPS:
        return x0
    return x0 * (np.exp(r * T) - 1.0) / (r * T)


def asian_level2_target(x0, r, T):
    """Analytic level-2 coordinate fed to the Asian analytic pipeline:
    ``E[int_0^T (X_t - X0) dt] = T (E[A] - X0)``.
    """
    return T * (asian_expected_average(x0, r, T) - x0)


def forward_closed_form(x0, K, r, T):
    """Closed-form forward price ``X0 - K e^{-rT}``."""
    return x0 - K * np.exp(-r * T)


def heston_integrated_variance(v0, kappa, theta, T):
    """``E[int_0^T v_t dt] = theta T + (v0 - theta)(1 - e^{-kappa T})/kappa``."""
    return theta * T + (v0 - theta) * (1.0 - np.exp(-kappa * T)) / kappa


def lognormal_moment(x0, r, sigma, T, j):
    """``E^Q[X_T^j]`` for GBM: ``X0^j exp(j(r-sigma^2/2)T + 1/2 j^2 sigma^2 T)``."""
    return x0 ** j * np.exp(j * (r - 0.5 * sigma ** 2) * T
                            + 0.5 * j * j * sigma ** 2 * T)


def power_claim_price(x0, r, sigma, T, p):
    """Closed-form price ``e^{-rT} E^Q[X_T^p]`` of the power claim ``X_T^p``."""
    return np.exp(-r * T) * lognormal_moment(x0, r, sigma, T, p)


def central_moment(x0, r, sigma, T, k):
    """``E^Q[(X_T - X0)^k]`` from the binomial expansion of the raw moments."""
    return sum(comb(k, j) * (-x0) ** (k - j) * lognormal_moment(x0, r, sigma, T, j)
               for j in range(k + 1))


def taylor_coeff(x0, p, k):
    """Coefficient ``c_k`` of ``(x - x0)^k`` in the Taylor series of ``x^p``
    about ``x0``: ``c_k = (p choose k) x0^{p-k}``."""
    num = 1.0
    for i in range(k):
        num *= (p - i)
    return num / factorial(k) * x0 ** (p - k)


def shuffle_power(letter, k):
    """``letter^{shuffle k}`` as a ``Counter`` of words (empty word for k=0)."""
    res = Counter({(): 1})
    for _ in range(k):
        nxt = Counter()
        for w, m in res.items():
            for w2, m2 in _shuffle.shuffle(w, tuple(letter)).items():
                nxt[w2] += m * m2
        res = nxt
    return res


def straight_line_signature(a, b, level):
    """Flat signature of the segment ``a -> b``: level-k term ``(b-a)^(x)k / k!``.

    Returned in iisignature word order (levels 1..level, empty word omitted).
    """
    inc = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    out = []
    fac = 1
    for k in range(1, level + 1):
        fac *= k
        term = inc
        for _ in range(k - 1):
            term = np.multiply.outer(term, inc)
        out.append((term / fac).ravel())
    return np.concatenate(out)


# --- covectors ---------------------------------------------------------------

def asian_covector(x0, K, T, n_channels):
    """Covector for the arithmetic-average Asian forward paying ``A - K``.

    ``f = (x0 - K)*(empty) + (1/T)*e_{(1,0)}`` (the word "price then time").
    Returns ``(const, vec)`` aligned with the flat signature, like
    ``price.forward_covector``.
    """
    const = x0 - K
    idx = signature.word_index((1, 0), n_channels)
    vec = np.zeros(idx + 1)
    vec[idx] = 1.0 / T
    return const, vec


# --- tensor algebra over flat (iisignature-ordered) signatures ---------------

def words_of_length(L, n_channels):
    """All words of length ``L`` over ``n_channels`` in lexicographic order
    (matching ``signature.word_index`` / iisignature)."""
    return list(itertools.product(range(n_channels), repeat=L))


def coeff(flat, word, n_channels):
    """Coefficient of ``word`` in a flat signature; empty word -> 1."""
    if len(word) == 0:
        return 1.0
    return flat[signature.word_index(word, n_channels)]


def tensor_mul_flat(A, B, n_channels, level):
    """Truncated tensor product of two flat signatures (levels 1..level).

    ``(A (x) B)_w = sum_{i} A_{w[:i]} B_{w[i:]}`` including empty-word splits,
    with the empty word carrying coefficient 1 (group-like elements).
    """
    out = []
    for L in range(1, level + 1):
        for w in words_of_length(L, n_channels):
            s = 0.0
            for i in range(L + 1):
                s += coeff(A, w[:i], n_channels) * coeff(B, w[i:], n_channels)
            out.append(s)
    return np.array(out)


def identity_flat(n_channels, level):
    """Flat signature of a constant (no-movement) path: all zeros."""
    n = sum(n_channels ** k for k in range(1, level + 1))
    return np.zeros(n)


# --- path generators ---------------------------------------------------------

def random_path(n_points, n_channels, seed, irregular=False):
    """A random piecewise-linear path, optionally on an irregular time grid.

    Channel 0 is a strictly increasing time channel; remaining channels are a
    random walk. Returns shape ``(n_points, n_channels)``.
    """
    rng = np.random.default_rng(seed)
    if irregular:
        gaps = rng.uniform(0.1, 1.0, size=n_points - 1)
        t = np.concatenate([[0.0], np.cumsum(gaps)])
    else:
        t = np.linspace(0.0, 1.0, n_points)
    cols = [t]
    for _ in range(n_channels - 1):
        cols.append(np.cumsum(rng.standard_normal(n_points)) * 0.3)
    return np.stack(cols, axis=-1)


def subdivide(path):
    """Insert the midpoint of every segment.

    The midpoints lie ON the existing piecewise-linear trajectory, so the
    geometric curve is unchanged (used by the reparameterisation test A5).
    """
    mids = 0.5 * (path[:-1] + path[1:])
    out = np.empty((2 * len(path) - 1, path.shape[1]))
    out[0::2] = path
    out[1::2] = mids
    return out
