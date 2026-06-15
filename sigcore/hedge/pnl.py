"""Simulate the shortfall distribution of a hedge.

Apply ``theta_t = <ell, S_{0,t}>`` on a (possibly coarse) rebalancing grid,
accumulate ``sum theta_k * dX`` as the trading P&L, and form the shortfall
``L = payoff - p0 - P&L`` per path. The payoff is the *true* liability on the
fine path; coarse rebalancing (or Heston) is what leaves a residual.
"""
from __future__ import annotations

import numpy as np

from . import strategy


def forward_payoff(times, price, K):
    return price[:, -1] - K


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def asian_payoff(times, price, K):
    """Trapezoidal arithmetic average minus K (matches the (lag,time) coord)."""
    avg = _trapz(price, times, axis=1) / (times[-1] - times[0])
    return avg - K


def _subsample(times, paths, rebalance_steps):
    n = times.size - 1
    if rebalance_steps is None or rebalance_steps >= n:
        return times, paths
    idx = np.linspace(0, n, rebalance_steps + 1).round().astype(int)
    idx = np.unique(idx)
    return times[idx], paths[:, idx]


def shortfall(ell_cov, payoff_fn, p0, times, paths, level, K,
              rebalance_steps=None):
    """Per-path shortfall ``L``.

    ``payoff_fn(times_fine, paths_fine, K)`` is evaluated on the fine grid;
    the trading P&L uses the (possibly coarse) rebalancing grid.
    """
    payoff = payoff_fn(times, paths, K)
    rt, rp = _subsample(times, paths, rebalance_steps)
    pnl = np.empty(rp.shape[0])
    for i in range(rp.shape[0]):
        theta = strategy.positions(ell_cov, rt, rp[i], level)
        pnl[i] = np.sum(theta[:-1] * np.diff(rp[i]))
    return payoff - p0 - pnl


def stats(L, tau=None):
    """Distribution summary: variance, mean, skew, CVaR95, optional P[L>tau]."""
    L = np.asarray(L)
    mean = L.mean()
    var = L.var(ddof=1)
    std = np.sqrt(var)
    skew = np.mean(((L - mean) / std) ** 3) if std > 0 else 0.0
    # CVaR95 of the LOSS: mean of the worst 5% (largest L = largest shortfall)
    q95 = np.quantile(L, 0.95)
    tail = L[L >= q95]
    cvar95 = tail.mean() if tail.size else q95
    out = {"mean": mean, "var": var, "std": std, "skew": skew,
           "cvar95": cvar95, "q95": q95}
    if tau is not None:
        out["p_exceed"] = float(np.mean(L > tau))
    return out
