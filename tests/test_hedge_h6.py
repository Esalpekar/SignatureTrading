"""H6 — asymmetric penalty reshapes the tail (THE headline).

Under Heston with coarse rebalancing (incomplete), fit the mean-variance hedge
and the convex-quartic hedge, apply BOTH to fresh paths, and show the
personalised risk profile trades variance for a thinner loss tail:
  Var(L_asym) >= Var(L_MV)  and  CVaR95(L_asym) < CVaR95(L_MV),  skew more negative.

Uses depth-0 (a constant position): there the hedge has genuine room to reshape
the tail. At depth-1 the Asian is nearly replicated under coarse rebalancing, so
the residual — and the room to trade variance for tail — collapses (verified in
calibration: depth-1 shows no MV/asym difference). Shortfall L>0 is a loss;
CVaR95 = mean of the worst 5%.
"""
import numpy as np

from sigcore import heston
from sigcore.hedge import embedding as emb, objective as obj, solve, pnl

X0, K, T = 1.0, 1.0, 1.0
HES = dict(v0=0.04, r=0.0, kappa=2.0, theta=0.04, xi=0.5, rho=-0.8)
P0 = X0 - K
DEPTH = 0
GAMMA, DELTA = 6.0, 14.5          # convex: delta >= 3 gamma^2/8 = 13.5
REBALANCE = 6


def _sim(n_paths, seed):
    return heston.simulate(X0, HES["v0"], HES["r"], HES["kappa"], HES["theta"],
                           HES["xi"], HES["rho"], T, 60, n_paths, seed=seed)


def _fit(penalty, n_paths=15_000, seed=0):
    payoff = emb.asian_covector(X0, K, T)
    basis = obj.strategy_basis(DEPTH)
    if penalty == "mv":
        level = obj.mean_variance_required_level(payoff, DEPTH)
        t, price, _ = _sim(n_paths, seed)
        es, _ = emb.expected_signature(t, price, level)
        A, b = obj.mean_variance_system(payoff, P0, basis, es)
        ell = solve.solve_mean_variance(A, b, lam=0.0)
    else:
        level = 4 * max(2, DEPTH + 1)          # quartic: maxlen(L) * 4
        t, price, _ = _sim(n_paths, seed)
        es, _ = emb.expected_signature(t, price, level)
        ell, _ = solve.newton(basis, payoff, P0, {2: 1.0, 3: GAMMA, 4: DELTA},
                              es, x0=np.zeros(len(basis)))
    return dict(zip(basis, ell))


def compute_h6(fit_seed=0, apply_seed=777, n_apply=40_000):
    """Shared by the report: fit MV and asymmetric hedges, apply to fresh paths,
    return the headline distribution stats for each."""
    ell_mv = _fit("mv", seed=fit_seed)
    ell_as = _fit("quartic", seed=fit_seed)
    t, price, _ = _sim(n_apply, seed=apply_seed)
    common = dict(times=t, paths=price, level=max(DEPTH, 1), K=K,
                  rebalance_steps=REBALANCE)
    L_mv = pnl.shortfall(ell_mv, pnl.asian_payoff, P0, **common)
    L_as = pnl.shortfall(ell_as, pnl.asian_payoff, P0, **common)
    tau = float(np.quantile(L_mv, 0.95))
    return {"mv": pnl.stats(L_mv, tau=tau), "as": pnl.stats(L_as, tau=tau)}


def test_H6_asymmetric_reshapes_tail():
    r = compute_h6()
    print("\n[H6] mean-variance vs asymmetric (Heston, coarse rebalance, depth 0):")
    print(f"   Var   : MV={r['mv']['var']:.3e}  asym={r['as']['var']:.3e}")
    print(f"   CVaR95: MV={r['mv']['cvar95']:.4f}  asym={r['as']['cvar95']:.4f}")
    print(f"   skew  : MV={r['mv']['skew']:.3f}  asym={r['as']['skew']:.3f}")
    print(f"   P[L>tau]: MV={r['mv']['p_exceed']:.3f}  asym={r['as']['p_exceed']:.3f}")

    assert r["as"]["var"] >= r["mv"]["var"] * 0.999       # accepts >= variance
    assert r["as"]["cvar95"] < r["mv"]["cvar95"]          # thinner loss tail
    assert r["as"]["skew"] < r["mv"]["skew"]              # more negative skew
