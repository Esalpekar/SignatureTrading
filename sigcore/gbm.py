"""Geometric Brownian motion under the risk-neutral measure.

Uses the *exact* GBM solution (not Euler-Maruyama), so the only sampling
error is Monte-Carlo, never discretisation of the SDE itself.
"""
from __future__ import annotations

import numpy as np


def simulate(x0, r, sigma, T, n_steps, n_paths, seed):
    """Simulate GBM with risk-neutral drift ``r``.

    ``X_{t+dt} = X_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)``,
    ``Z ~ N(0, 1)`` i.i.d. This is the exact solution of GBM.

    Returns ``(times, paths)`` with ``times`` shape ``(n_steps+1,)`` and
    ``paths`` shape ``(n_paths, n_steps+1)``. Deterministic given ``seed``.
    """
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))

    drift = (r - 0.5 * sigma * sigma) * dt
    increments = drift + sigma * np.sqrt(dt) * Z          # log-increments
    log_paths = np.cumsum(increments, axis=1)
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), log_paths], axis=1
    )
    paths = x0 * np.exp(log_paths)
    return times, paths


def simulate_brownian(sigma, T, n_steps, n_paths, seed):
    """Simulate driftless Brownian motion ``W`` directly (W_0 = 0).

    ``W_{t+dt} = W_t + sigma*sqrt(dt)*Z``. Needed for T1, whose analytic
    target (Fawcett's formula) is stated for the Brownian driver ``(t, W_t)``;
    GBM with ``x0=0`` would be degenerate. Deterministic given ``seed``.
    """
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))

    increments = sigma * np.sqrt(dt) * Z
    W = np.cumsum(increments, axis=1)
    W = np.concatenate([np.zeros((n_paths, 1)), W], axis=1)
    return times, W


def simulate_correlated_brownian(rho, sigma, T, n_steps, n_paths, seed):
    """Two driftless Brownian motions ``(W1, W2)`` with correlation ``rho``.

    Built from independent Gaussians via ``Z2 = rho*Z1 + sqrt(1-rho^2)*Zind``.
    Returns ``(times, W1, W2)`` with ``W1``/``W2`` shape ``(n_paths, n_steps+1)``.
    Needed for the multi-asset joint-signature oracle (cross-channel Levy area).
    Deterministic given ``seed``.
    """
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_steps))
    Z_ind = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + np.sqrt(1.0 - rho * rho) * Z_ind

    def _accumulate(Z):
        W = np.cumsum(sigma * np.sqrt(dt) * Z, axis=1)
        return np.concatenate([np.zeros((n_paths, 1)), W], axis=1)

    return times, _accumulate(Z1), _accumulate(Z2)


def _correlated_increments(corr, n_steps, n_paths, seed):
    """Correlated standard-normal increments: shape ``(n_paths, n_steps, M)``."""
    corr = np.asarray(corr, dtype=float)
    M = corr.shape[0]
    chol = np.linalg.cholesky(corr)                      # lower-triangular
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps, M))
    return Z @ chol.T                                    # corr(., .) = corr


def simulate_correlated_brownian_multi(corr, T, n_steps, n_paths, seed):
    """``M`` driftless unit Brownian motions with correlation matrix ``corr``.

    Returns ``(times, W)`` with ``W`` a list of ``M`` arrays shape
    ``(n_paths, n_steps+1)``. For the joint-signature engine oracle (M0).
    """
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    dW = _correlated_increments(corr, n_steps, n_paths, seed) * np.sqrt(dt)
    M = np.asarray(corr).shape[0]
    W = []
    for m in range(M):
        w = np.cumsum(dW[:, :, m], axis=1)
        W.append(np.concatenate([np.zeros((n_paths, 1)), w], axis=1))
    return times, W


def simulate_correlated_gbm(x0, r, sigma, corr, T, n_steps, n_paths, seed):
    """``M``-asset GBM with correlated drivers (exact log-Euler-free recursion).

    ``x0``, ``sigma`` are length-``M``; ``corr`` is ``M x M``. Returns
    ``(times, X)`` with ``X`` a list of ``M`` arrays shape ``(n_paths, n_steps+1)``.
    """
    x0 = np.asarray(x0, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)
    dW = _correlated_increments(corr, n_steps, n_paths, seed) * np.sqrt(dt)
    M = x0.size
    X = []
    for m in range(M):
        incr = (r - 0.5 * sigma[m] ** 2) * dt + sigma[m] * dW[:, :, m]
        logp = np.cumsum(incr, axis=1)
        logp = np.concatenate([np.zeros((n_paths, 1)), logp], axis=1)
        X.append(x0[m] * np.exp(logp))
    return times, X
