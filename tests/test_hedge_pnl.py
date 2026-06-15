"""H3 — variance reduction (monotone with depth, plateau at the floor);
H8 — out-of-sample consistency.

Incompleteness for these linear claims comes from coarse rebalancing (and the
signature hedge being fit to the average E[S], not each path) under Heston
stochastic vol. Run at r=0 (martingale, mean-zero shortfall).
"""
import numpy as np

from sigcore import gbm, heston
from sigcore.hedge import embedding as emb, objective as obj, solve, pnl
import helpers as H

X0, K, T = 1.0, 1.0, 1.0
# Heston, r=0 (Feller-satisfying)
HES = dict(v0=0.04, r=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
P0 = X0 - K                      # E[A] = X0 at r=0


def _fit_mv(depth, n_paths=30_000, seed=0):
    payoff = emb.asian_covector(X0, K, T)
    level = obj.mean_variance_required_level(payoff, depth)
    t, price, _ = heston.simulate(X0, HES["v0"], HES["r"], HES["kappa"],
                                  HES["theta"], HES["xi"], HES["rho"],
                                  T, 60, n_paths, seed=seed)
    es, _ = emb.expected_signature(t, price, level)
    basis = obj.strategy_basis(depth)
    A, b = obj.mean_variance_system(payoff, P0, basis, es)
    ell = solve.solve_mean_variance(A, b, lam=0.0)
    return dict(zip(basis, ell))


def _apply(ell_cov, depth, rebalance_steps, n_paths=20_000, seed=99):
    t, price, _ = heston.simulate(X0, HES["v0"], HES["r"], HES["kappa"],
                                  HES["theta"], HES["xi"], HES["rho"],
                                  T, 60, n_paths, seed=seed)
    return pnl.shortfall(ell_cov, pnl.asian_payoff, P0, t, price,
                         level=max(depth, 1), K=K, rebalance_steps=rebalance_steps)


def test_H3_variance_reduction_with_depth():
    rebalance = 12                       # coarse -> genuine residual
    L_unhedged = _apply({}, 0, rebalance)
    var_un = L_unhedged.var(ddof=1)

    variances = []
    for depth in (0, 1, 2):
        ell = _fit_mv(depth)
        L = _apply(ell, depth, rebalance)
        variances.append(L.var(ddof=1))
    print(f"\n[H3] Var unhedged={var_un:.3e}  by depth: "
          + "  ".join(f"d{d}={v:.3e}" for d, v in enumerate(variances)))

    assert variances[0] < var_un                 # hedging helps
    # monotone non-increasing with depth (small slack for MC noise)
    assert variances[1] <= variances[0] * 1.02
    assert variances[2] <= variances[1] * 1.02
    assert variances[-1] > 0                      # incompleteness floor


def test_H8_out_of_sample_consistency():
    # Fit ell* on E[S]; then compare the realised shortfall distribution on the
    # FIT paths (in-sample) vs FRESH paths (out-of-sample), at the same coarse
    # rebalancing resolution. In the stationary generated-data regime in-sample
    # = out-of-sample, so a mismatch means the feedback eval theta_t=<ell,S_{0,t}>
    # (strategy.py) is wired wrong. Coarse grid keeps L non-degenerate.
    depth, rebalance = 1, 12
    ell = _fit_mv(depth, n_paths=30_000, seed=3)

    L_in = _apply(ell, depth, rebalance, n_paths=30_000, seed=3)    # in-sample paths
    L_out = _apply(ell, depth, rebalance, n_paths=30_000, seed=99)  # fresh paths
    m_in = np.mean(L_in ** 2)
    m_out = np.mean(L_out ** 2)
    se = np.sqrt(np.var(L_in ** 2, ddof=1) / L_in.size
                 + np.var(L_out ** 2, ddof=1) / L_out.size)
    print(f"\n[H8] in-sample E[L^2]={m_in:.5e}  out-of-sample={m_out:.5e}  "
          f"|err|={abs(m_in-m_out):.2e}  4SE={4*se:.2e}")
    assert abs(m_in - m_out) <= 4 * se
