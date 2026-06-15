"""MH4 — conditioning and the pairs/ridge tension (the subtle gate).

(a) cond(A) grows with correlation and with M (highly correlated assets are
near-redundant instruments -> near-singular A). (b) ridge keeps the solve
bounded across seeds. (c) CRITICAL: the spread (+1,-1) pairs position must
survive ridging -- the cross-asset signal lives in the near-null-space direction
ridge damps, so this guards against quietly ridging the relative-value away.
"""
import numpy as np

from sigcore import gbm
from sigcore.hedge import multiasset as ma, solve, scaling

T = 1.0
X0 = [1.0, 1.0]
SIG = [0.2, 0.25]


def _spread_system(rho, depth=0, n_paths=60_000, seed=0):
    corr = np.array([[1.0, rho], [rho, 1.0]])
    t, X = gbm.simulate_correlated_gbm(X0, 0.0, SIG, corr, T, 50, n_paths, seed=seed)
    es, _ = ma.expected_signature(t, X, 2 * (depth + 1))
    payoff = ma.spread_forward_covector(X0, 0.0, 0, 1, 2)
    basis = ma.strategy_basis(2, depth)
    A, b, idx = ma.mean_variance_system(payoff, 0.0, basis, 2, es)
    return A, b, idx, basis


def test_MH4a_conditioning_grows_with_correlation():
    conds = []
    for rho in (0.0, 0.6, 0.95):
        A, _, _, _ = _spread_system(rho)
        conds.append(scaling.condition_numbers(A)[0])
    print(f"\n[MH4a] cond(A) by rho (0, .6, .95): "
          + "  ".join(f"{c:.1e}" for c in conds))
    assert conds[2] > conds[1] > conds[0]            # more correlation -> worse


def test_MH4b_ridge_bounded_across_seeds():
    rho = 0.9
    ridged = []
    for seed in range(20):
        A, b, idx, basis = _spread_system(rho, n_paths=6_000, seed=100 + seed)
        ridged.append(scaling.ridged_solve(A, b, lam=1e-3))
    ridged = np.array(ridged)
    print(f"\n[MH4b] rho=.9: ridged coeff std={ridged.std(0).max():.2e}  "
          f"max|coef|={np.abs(ridged).max():.2f}")
    assert ridged.std(0).max() < 0.5 and np.abs(ridged).max() < 5.0


def test_MH4c_ridge_preserves_pairs_position():
    # Highly correlated -> near-singular A; the spread (+1,-1) lives in the
    # near-null-space. A small ridge must stabilise WITHOUT erasing it.
    A, b, idx, basis = _spread_system(0.95)
    ell = scaling.ridged_solve(A, b, lam=1e-3)
    th = [ell[idx[(m, ())]] for m in range(2)]
    print(f"\n[MH4c] rho=.95 ridged theta=({th[0]:+.3f}, {th[1]:+.3f})  target (+1,-1)")
    assert abs(th[0] - 1.0) < 5e-2 and abs(th[1] + 1.0) < 5e-2

    # contrast: over-ridging DOES erase the pairs position (documents the tension)
    ell_over = scaling.ridged_solve(A, b, lam=10.0)
    th_over = [ell_over[idx[(m, ())]] for m in range(2)]
    print(f"        over-ridged (lam=10) theta=({th_over[0]:+.3f}, {th_over[1]:+.3f}) "
          f"-> shrunk, pairs lost")
    assert abs(th_over[0]) < 0.5            # over-ridge shrinks the position
