"""Heston stochastic-volatility simulator.

``dX = r X dt + sqrt(v) X dW1``
``dv = kappa (theta - v) dt + xi sqrt(v) dW2``,  ``corr(dW1, dW2) = rho``.

Discretised with **full-truncation Euler**: the variance is clamped at 0 in
both the drift/diffusion evaluation and the stored path, so variance is never
negative even when the Feller condition ``2 kappa theta >= xi^2`` is violated
(required for D5). The log-price is integrated exactly given the variance, so
the price stays positive. Deterministic given ``seed``.
"""
from __future__ import annotations

import numpy as np


def simulate(x0, v0, r, kappa, theta, xi, rho, T, n_steps, n_paths, seed):
    """Simulate Heston price and variance paths.

    Returns ``(times, price, variance)`` with ``times`` shape ``(n_steps+1,)``
    and ``price``/``variance`` shape ``(n_paths, n_steps+1)``.
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0.0, T, n_steps + 1)

    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_steps))
    Zind = rng.standard_normal((n_paths, n_steps))
    Z2 = rho * Z1 + np.sqrt(1.0 - rho * rho) * Zind   # corr(Z1, Z2) = rho

    price = np.empty((n_paths, n_steps + 1))
    variance = np.empty((n_paths, n_steps + 1))
    price[:, 0] = x0
    variance[:, 0] = v0

    log_x = np.full(n_paths, np.log(x0))
    v = np.full(n_paths, float(v0))
    for k in range(n_steps):
        v_pos = np.maximum(v, 0.0)            # full truncation
        sqrt_v = np.sqrt(v_pos)
        log_x += (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * Z1[:, k]
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * Z2[:, k]
        v = np.maximum(v, 0.0)                # stored variance stays >= 0
        price[:, k + 1] = np.exp(log_x)
        variance[:, k + 1] = v

    return times, price, variance
