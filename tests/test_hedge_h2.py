"""H2 — level-0 hedge = regression coefficient Cov(F, dX)/Var(dX).

Run at r=0 so dX is a martingale increment (E[dX]=0): then the mean-variance
E[L^2] optimum equals the variance-minimising Cov/Var ratio, and the forward's
depth-0 ratio is exactly 1. p0 = E[F]. Oracle is a direct-MC covariance ratio
that shares none of the signature machinery. Within 4*SE.
"""
import numpy as np
import pytest

from sigcore import gbm
from sigcore.hedge import embedding as emb, objective as obj
import helpers as H

X0, R, SIGMA, T, K = 1.0, 0.0, 0.2, 1.0, 1.0


def _depth0_ratio(payoff_cov, p0, level):
    t, paths = gbm.simulate(X0, R, SIGMA, T, 50, 100_000, seed=0)
    es, _ = emb.expected_signature(t, paths, level)
    A, b = obj.mean_variance_system(payoff_cov, p0, obj.strategy_basis(0), es)
    return b[0] / A[0, 0]


def _direct_cov_ratio(payoff_fn, seed=1):
    t, paths = gbm.simulate(X0, R, SIGMA, T, 50, 200_000, seed=seed)
    F = payoff_fn(t, paths)
    dX = paths[:, -1] - paths[:, 0]
    ratio = np.cov(F, dX)[0, 1] / np.var(dX, ddof=1)
    # delta-method SE of the ratio via bootstrap-free linearisation
    n = F.size
    resid = (F - F.mean()) - ratio * (dX - dX.mean())
    se = np.sqrt(np.mean(resid ** 2) / (n * np.var(dX, ddof=1)))
    return ratio, se


def test_H2_forward_depth0():
    p0 = X0 - K                       # E[F] at r=0
    ell0 = _depth0_ratio(emb.forward_covector(X0, K), p0, level=2)
    ratio, se = _direct_cov_ratio(lambda t, p: p[:, -1] - K)
    print(f"\n[H2 forward] ell0={ell0:.5f}  direct Cov/Var={ratio:.5f}  "
          f"(=1 exact)  4SE={4*se:.1e}")
    assert abs(ell0 - 1.0) < 1e-3            # forward replicates: ratio = 1
    assert abs(ell0 - ratio) <= 4 * se + 1e-9   # forward is exact: SE ~ 0


def test_H2_asian_depth0():
    from sigcore.hedge.pnl import asian_payoff
    p0 = X0 - K                       # E[A] = X0 at r=0
    ell0 = _depth0_ratio(emb.asian_covector(X0, K, T), p0, level=4)
    ratio, se = _direct_cov_ratio(lambda t, p: asian_payoff(t, p, K))
    print(f"\n[H2 asian] ell0={ell0:.5f}  direct Cov/Var={ratio:.5f}  4SE={4*se:.1e}")
    assert abs(ell0 - ratio) <= 4 * se
