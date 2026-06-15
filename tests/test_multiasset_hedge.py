"""MH0-MH5 — multi-asset hedging gates.

Replication/regression/structure gates run under complete correlated GBM (r=0
for the clean martingale oracles); the cross-asset story lives in the
off-diagonal blocks, which are exercised directly (MH3) and via a pure
cross-hedge (MH5).
"""
import numpy as np
import iisignature
import pytest

from sigcore import gbm, signature
from sigcore.hedge import multiasset as ma, covector as cov, solve, scaling
import helpers as H

T = 1.0
CORR = np.array([[1.0, 0.6], [0.6, 1.0]])
IDENT = np.eye(2)
X0 = [1.0, 1.0]
SIG = [0.2, 0.25]
GAMMA, DELTA = 0.5, 0.2


def _es(corr, level, r=0.0, n_paths=60_000, seed=0, sig=SIG):
    t, X = gbm.simulate_correlated_gbm(X0, r, sig, corr, T, 50, n_paths, seed=seed)
    return ma.expected_signature(t, X, level)[0], (t, X)


def basket_payoff(weights, K):
    return lambda times, paths: sum(w * paths[i][:, -1] for i, w in enumerate(weights)) - K


def spread_payoff(K):
    return lambda times, paths: paths[0][:, -1] - paths[1][:, -1] - K


# ---- MH0 — objective cross-check (bedrock) ----------------------------------
def test_MH0_objective_vs_direct_mc():
    M, depth = 2, 1
    basis = ma.strategy_basis(M, depth)
    rng = np.random.default_rng(0)
    ell = [{w: float(v) for w, v in zip(basis, rng.standard_normal(len(basis)))}
           for _ in range(M)]
    payoff = ma.basket_forward_covector(X0, [1.0, 0.5], 1.0, M)
    p0 = 0.5                                   # arbitrary premium
    loss = ma.loss_covector(payoff, p0, ell, M)
    level = 4                                  # MV/quartic-depth0 fit feasibly here

    # per-path exact identities (no MC noise): literal P&L vs shuffle side
    t, X = gbm.simulate_correlated_gbm(X0, 0.03, SIG, CORR, T, 30, 200, seed=7)
    sigs = iisignature.sig(np.ascontiguousarray(ma.enlarged_batch(t, X)), level)
    payoff_real = basket_payoff([1.0, 0.5], 1.0)(t, X)
    lay = ma.layout(M)
    trade = cov.add(*[cov.append_letter(ell[m], lay["leads"][m]) for m in range(M)])
    max_pnl = max_loss = 0.0
    for p in range(X[0].shape[0]):
        theta = ma.positions(ell, t, [X[i][p] for i in range(M)], level, M)
        pnl_direct = sum(np.sum(theta[m, :-1] * np.diff(X[m][p])) for m in range(M))
        pnl_cov = cov.contract(trade, sigs[p], lay["d"])
        max_pnl = max(max_pnl, abs(pnl_direct - pnl_cov))
        L_direct = payoff_real[p] - p0 - pnl_direct
        L_cov = cov.contract(loss, sigs[p], lay["d"])
        max_loss = max(max_loss, abs(L_direct - L_cov))
    print(f"\n[MH0] per-path err: P&L={max_pnl:.1e}  loss={max_loss:.1e}")
    assert max_pnl < 1e-8 and max_loss < 1e-8

    # 4*SE objective cross-check (independent samples), P = x^2
    es, _ = _es(CORR, level, r=0.03, n_paths=40_000, seed=11)
    R_shuffle = ma.objective_value(loss, {2: 1.0}, es, M)
    L = ma.shortfall(ell, basket_payoff([1.0, 0.5], 1.0), p0,
                     *gbm.simulate_correlated_gbm(X0, 0.03, SIG, CORR, T, 50, 40_000, seed=22),
                     level, M)
    direct, se = np.mean(L ** 2), np.std(L ** 2, ddof=1) / np.sqrt(L.size)
    print(f"     shuffle={R_shuffle:.4f} direct={direct:.4f} band={4*np.sqrt(2)*se:.1e}")
    assert abs(R_shuffle - direct) <= 4 * np.sqrt(2) * se


# ---- MH1 — cross-asset replication oracles (r=0) ----------------------------
def _depth0_fit(payoff, p0, corr, level=2):
    es, _ = _es(corr, level, r=0.0, n_paths=60_000, seed=0)
    basis = ma.strategy_basis(2, 0)
    A, b, idx = ma.mean_variance_system(payoff, p0, basis, 2, es)
    ell_vec = solve.solve_mean_variance(A, b, 0.0)
    return ell_vec, idx, basis


def test_MH1_spread_and_basket_replication():
    M = 2
    # spread X1 - X2 - K -> theta = (+1, -1)
    spread = ma.spread_forward_covector(X0, 0.0, 0, 1, M)
    ellv, idx, basis = _depth0_fit(spread, X0[0] - X0[1] - 0.0, CORR)
    th = [ellv[idx[(m, ())]] for m in range(M)]
    print(f"\n[MH1 spread] theta = ({th[0]:+.3f}, {th[1]:+.3f})  target (+1, -1)")
    assert abs(th[0] - 1.0) < 1e-2 and abs(th[1] + 1.0) < 1e-2

    # basket 0.7 X1 + 0.3 X2 - K -> theta = (0.7, 0.3)
    w = [0.7, 0.3]
    basket = ma.basket_forward_covector(X0, w, 1.0, M)
    p0 = float(np.dot(w, X0) - 1.0)
    ellv, idx, basis = _depth0_fit(basket, p0, CORR)
    thb = [ellv[idx[(m, ())]] for m in range(M)]
    print(f"[MH1 basket] theta = ({thb[0]:.3f}, {thb[1]:.3f})  target (0.7, 0.3)")
    assert abs(thb[0] - 0.7) < 1e-2 and abs(thb[1] - 0.3) < 1e-2

    # E[L^2] ~ 0 for the exact constant hedge (spread)
    ell_const = [{(): 1.0}, {(): -1.0}]
    t, X = gbm.simulate_correlated_gbm(X0, 0.0, SIG, CORR, T, 50, 5_000, seed=3)
    L = ma.shortfall(ell_const, spread_payoff(0.0), 0.0, t, X, 1, M)
    print(f"[MH1 spread] constant (+1,-1) max|L| = {np.max(np.abs(L)):.1e}")
    assert np.max(np.abs(L)) < 1e-9


# ---- MH2 — level-0 = multivariate regression --------------------------------
def test_MH2_multivariate_regression():
    M = 2
    w = [1.0, 0.5]
    basket = ma.basket_forward_covector(X0, w, 1.0, M)
    p0 = float(np.dot(w, X0) - 1.0)             # = E[F] at r=0
    ellv, idx, _ = _depth0_fit(basket, p0, CORR)
    ell0 = np.array([ellv[idx[(m, ())]] for m in range(M)])

    # oracle: Sigma_dX^{-1} Cov(F, dX) via direct MC, independent seed
    t, X = gbm.simulate_correlated_gbm(X0, 0.0, SIG, CORR, T, 50, 200_000, seed=5)
    F = basket_payoff(w, 1.0)(t, X)
    dX = np.stack([X[m][:, -1] - X[m][:, 0] for m in range(M)], axis=1)
    Sigma = np.cov(dX.T)
    cross = np.array([np.cov(F, dX[:, m])[0, 1] for m in range(M)])
    oracle = np.linalg.solve(Sigma, cross)
    print(f"\n[MH2] ell0={np.round(ell0,4)}  regression oracle={np.round(oracle,4)}")
    assert np.max(np.abs(ell0 - oracle)) < 5e-2


# ---- MH3 — off-diagonal blocks vanish iff independence -----------------------
def _block_offdiag_norm(A, idx, basis, M):
    nb = len(basis)
    off = 0.0
    for m in range(M):
        for n in range(M):
            if m == n:
                continue
            block = A[m * nb:(m + 1) * nb, n * nb:(n + 1) * nb]
            off = max(off, np.max(np.abs(block)))
    return off


def test_MH3_offdiagonal_blocks_track_dependence():
    M, depth = 2, 1
    basis = ma.strategy_basis(M, depth)
    w = [1.0, 0.5]
    payoff = ma.basket_forward_covector(X0, w, 1.0, M)
    p0 = float(np.dot(w, X0) - 1.0)

    es_i, _ = _es(IDENT, 4, r=0.0, n_paths=60_000, seed=0)
    es_c, _ = _es(CORR, 4, r=0.0, n_paths=60_000, seed=0)
    Ai, bi, idx = ma.mean_variance_system(payoff, p0, basis, M, es_i)
    Ac, bc, _ = ma.mean_variance_system(payoff, p0, basis, M, es_c)

    nb = len(basis)
    diag_scale = max(np.max(np.abs(np.diag(Ai))), np.max(np.abs(np.diag(Ac))))
    off_i = _block_offdiag_norm(Ai, idx, basis, M) / diag_scale
    off_c = _block_offdiag_norm(Ac, idx, basis, M) / diag_scale
    # joint vs decoupled solve
    ell_joint = solve.solve_mean_variance(Ac, bc, 0.0)
    ell_dec = ell_joint.copy()
    for m in range(M):
        sl = slice(m * nb, (m + 1) * nb)
        ell_dec[sl] = solve.solve_mean_variance(Ac[sl, sl], bc[sl], 0.0)
    diff = np.max(np.abs(ell_joint - ell_dec))
    print(f"\n[MH3] off-diag/diag: independent={off_i:.2e}  correlated={off_c:.2e}  "
          f"joint-vs-decoupled diff={diff:.3f}")
    assert off_i < 1e-2                       # independent -> block-diagonal
    assert off_c > 0.1                        # correlated -> materially nonzero
    assert diff > 1e-2                        # joint solve differs from decoupled


# ---- MH5 — cross-hedge value (hedge X1 using only X2) -----------------------
def _cross_hedge_var(corr, rho_label, n_paths=30_000):
    M = 2
    # liability on X1 (forward), but allow trading only X2: zero out asset-0 basis
    payoff = ma.basket_forward_covector(X0, [1.0, 0.0], 1.0, M)
    p0 = X0[0] - 1.0
    basis = ma.strategy_basis(M, 1)
    es, _ = _es(corr, 4, r=0.0, n_paths=n_paths, seed=0)
    A, b, idx = ma.mean_variance_system(payoff, p0, basis, M, es)
    # restrict to asset-1 trades only (cross-hedge): drop asset-0 rows/cols
    keep = [idx[(1, w)] for w in basis]
    ell2 = solve.solve_mean_variance(A[np.ix_(keep, keep)], b[keep], 0.0)
    ell = [{w: 0.0 for w in basis}, dict(zip(basis, ell2))]
    t, X = gbm.simulate_correlated_gbm(X0, 0.0, SIG, corr, T, 50, n_paths, seed=9)
    L = ma.shortfall(ell, basket_payoff([1.0, 0.0], 1.0), p0, t, X, 1, M)
    L0 = basket_payoff([1.0, 0.0], 1.0)(t, X) - p0
    return L0.var(ddof=1), L.var(ddof=1), np.max(np.abs(ell2))


def test_MH5_cross_hedge_value():
    var_un, var_hi, coef_hi = _cross_hedge_var(np.array([[1.0, 0.8], [0.8, 1.0]]), "0.8")
    _, var_lo, _ = _cross_hedge_var(np.array([[1.0, 0.4], [0.4, 1.0]]), "0.4")
    _, var_ind, coef_ind = _cross_hedge_var(IDENT, "0.0")
    print(f"\n[MH5] Var unhedged={var_un:.3e}  cross-hedge rho=.8 {var_hi:.3e}  "
          f"rho=.4 {var_lo:.3e}  indep {var_ind:.3e}  (indep coef={coef_ind:.2e})")
    assert var_hi < var_un and var_lo < var_un       # cross-hedge helps iff rho!=0
    assert var_hi < var_lo                           # more correlation -> more help
    assert var_ind > var_un * 0.98                   # independent -> no help (functional proof)
    assert coef_ind < 0.05 and coef_ind < coef_hi    # indep theta2 ~ 0 (MC noise) << correlated
