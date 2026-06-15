"""H1 — complete-market replication oracle; H5 — convexity / unique optimum."""
import numpy as np
import pytest

from sigcore import gbm
from sigcore.hedge import embedding as emb, objective as obj, solve, pnl
import helpers as H

GAMMA, DELTA = 0.5, 0.2
QUARTIC = {2: 1.0, 3: GAMMA, 4: DELTA}


def test_H1_complete_market_replication():
    # GBM, r=0, forward, mean-variance, lambda ~ 0: the unique known exact hedge
    # is the constant strategy theta=1, which gives L=0 per path.
    x0, r, sigma, T, K = 1.0, 0.0, 0.2, 1.0, 1.0
    basis = obj.strategy_basis(1)                         # [(), time, lag]
    payoff = emb.forward_covector(x0, K)
    p0 = H.forward_closed_form(x0, K, r, T)               # = x0 - K = 0

    t, paths = gbm.simulate(x0, r, sigma, T, 50, 50_000, seed=0)
    es, _ = emb.expected_signature(t, paths, level=4)
    A, b = obj.mean_variance_system(payoff, p0, basis, es)
    ell = solve.solve_mean_variance(A, b, lam=0.0)

    print(f"\n[H1] ell* = {dict(zip(basis, np.round(ell, 4)))}")
    # ell* ~ constant strategy theta = 1 (coeff 1 on empty word, ~0 elsewhere)
    assert abs(ell[0] - 1.0) < 1e-2
    assert np.max(np.abs(ell[1:])) < 1e-2

    # the exact constant-1 hedge drives L to 0 per path (machine precision)
    ell_const = {(): 1.0}
    tf, pf = gbm.simulate(x0, r, sigma, T, 50, 5_000, seed=1)
    L = pnl.shortfall(ell_const, pnl.forward_payoff, p0, tf, pf, level=2, K=K)
    print(f"     constant-1 hedge: max|L| = {np.max(np.abs(L)):.2e}")
    assert np.max(np.abs(L)) < 1e-9


def test_H5_convex_unique_optimum():
    # Convex quartic, forward, depth 1: Newton from several starts -> same ell*.
    x0, r, sigma, T, K = 1.0, 0.03, 0.3, 1.0, 1.0
    basis = obj.strategy_basis(1)
    payoff = emb.forward_covector(x0, K)
    p0 = H.forward_closed_form(x0, K, r, T)

    t, paths = gbm.simulate(x0, r, sigma, T, 24, 8_000, seed=0)
    es, _ = emb.expected_signature(t, paths, level=8)      # quartic depth1 -> lvl 8

    sols = []
    rng = np.random.default_rng(0)
    for s in range(4):
        x0_start = (rng.standard_normal(len(basis)) * 2.0) if s else np.zeros(len(basis))
        ell, info = solve.newton(basis, payoff, p0, QUARTIC, es, x0=x0_start)
        sols.append(ell)
    spread = max(np.max(np.abs(s - sols[0])) for s in sols[1:])
    print(f"\n[H5] quartic Newton spread across 4 starts: {spread:.2e}")
    assert spread < 1e-6

    # mean-variance is the one-Newton-step special case (constant Hessian).
    A, b = obj.mean_variance_system(payoff, p0, basis, es)
    ell_lin = solve.solve_mean_variance(A, b, lam=0.0)
    ell_newt, info = solve.newton(basis, payoff, p0, {2: 1.0}, es, x0=np.zeros(len(basis)))
    print(f"     MV: linear vs Newton match {np.max(np.abs(ell_lin-ell_newt)):.2e}, "
          f"iters={info['iters']}")
    assert np.max(np.abs(ell_lin - ell_newt)) < 1e-8
    assert info["iters"] <= 2
