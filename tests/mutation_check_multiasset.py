"""Mutation testing for the multi-asset gates (spec section 4).

Inject a real bug, confirm the appropriate gate goes RED. Run:
  python tests/mutation_check_multiasset.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import iisignature

from sigcore import gbm, signature
from sigcore.hedge import multiasset as ma, covector as cov, solve, scaling
import helpers as H

T = 1.0
X0 = [1.0, 1.0]
SIG = [0.2, 0.25]
CORR = np.array([[1.0, 0.7], [0.7, 1.0]])
results = []


def report(name, gate, caught):
    results.append(caught)
    print(f"  [{'CAUGHT' if caught else 'LEAK  '}] {name:38s} -> {gate}")


# ---- M-swap: swap trade letters a_1 <-> a_2 ---------------------------------
def m_swap_trade_letters():
    # MH0 per-path: literal P&L (correct asset) vs shuffle side with swapped a.
    M = 2
    basis = ma.strategy_basis(M, 1)
    rng = np.random.default_rng(0)
    ell = [{w: float(v) for w, v in zip(basis, rng.standard_normal(len(basis)))}
           for _ in range(M)]
    payoff = ma.basket_forward_covector(X0, [1.0, 0.5], 1.0, M)
    p0 = 0.5
    t, X = gbm.simulate_correlated_gbm(X0, 0.03, SIG, CORR, T, 30, 100, seed=7)
    sigs = iisignature.sig(np.ascontiguousarray(ma.enlarged_batch(t, X)), 4)
    lay = ma.layout(M)
    leads_swapped = [lay["leads"][1], lay["leads"][0]]      # a_1 <-> a_2
    trade_bad = cov.add(*[cov.append_letter(ell[m], leads_swapped[m]) for m in range(M)])
    err = 0.0
    for p in range(X[0].shape[0]):
        theta = ma.positions(ell, t, [X[i][p] for i in range(M)], 4, M)
        pnl_direct = sum(np.sum(theta[m, :-1] * np.diff(X[m][p])) for m in range(M))
        pnl_cov_bad = cov.contract(trade_bad, sigs[p], lay["d"])
        err = max(err, abs(pnl_direct - pnl_cov_bad))
    report("swap trade letters a1<->a2", f"MH0 P&L err={err:.1e}", err > 1e-6)


# ---- M-zerocross: zero the cross-asset coordinates --------------------------
def m_zero_cross():
    # M0: zeroing the level-2 cross coords makes the cross coordinate != rho*T.
    t, W = gbm.simulate_correlated_brownian_multi(CORR, T, 200, 60_000, seed=0)
    emb = H.time_augment_multi(t, *W)
    sigs = iisignature.sig(np.ascontiguousarray(emb), 3)
    mc = sigs.mean(0)
    d = 3
    i12, i21 = signature.word_index((1, 2), d), signature.word_index((2, 1), d)
    mc_bad = mc.copy()
    mc_bad[i12] = mc_bad[i21] = 0.0                        # zero cross coords
    sym = mc_bad[i12] + mc_bad[i21]
    report("zero cross coords", f"M0 sym={sym:.3f} (target {CORR[0,1]*T})",
           abs(sym - CORR[0, 1] * T) > 1e-2)

    # MH5: zero the MIXED-channel coordinates of the joint E[S] (the cross-asset
    # story flows through b here, not A's off-diagonal) -> cross-hedge vanishes.
    payoff = ma.basket_forward_covector(X0, [1.0, 0.0], 1.0, 2)
    p0 = X0[0] - 1.0
    basis = ma.strategy_basis(2, 0)              # depth-0: cross-hedge via b only
    es, _ = ma.expected_signature(*_csim(30_000, 0), 3)
    es_bad = _zero_cross_channel_coords(es, level=3, d=5,
                                        group0={1, 2}, group1={3, 4})
    A, b, idx = ma.mean_variance_system(payoff, p0, basis, 2, es_bad)
    keep = [idx[(1, w)] for w in basis]
    ell2 = solve.solve_mean_variance(A[np.ix_(keep, keep)], b[keep], 0.0)
    t, X = gbm.simulate_correlated_gbm(X0, 0.0, SIG, CORR, T, 50, 30_000, seed=9)
    ell = [{w: 0.0 for w in basis}, dict(zip(basis, ell2))]
    L = ma.shortfall(ell, lambda tt, xx: xx[0][:, -1] - 1.0, p0, t, X, 1, 2)
    L0 = X[0][:, -1] - 1.0 - p0
    report("zero cross coords", f"MH5 var {L0.var():.3e}->{L.var():.3e}",
           L.var(ddof=1) > L0.var(ddof=1) * 0.98)


def _zero_cross_channel_coords(es, level, d, group0, group1):
    import itertools
    out = es.copy()
    for L in range(1, level + 1):
        for w in itertools.product(range(d), repeat=L):
            sw = set(w)
            if (sw & group0) and (sw & group1):           # mixes both assets
                out[signature.word_index(w, d)] = 0.0
    return out


def _csim(n, seed):
    return gbm.simulate_correlated_gbm(X0, 0.0, SIG, CORR, T, 50, n, seed=seed)


# ---- M-negate-crossvar: negate the cross-variation recovery -----------------
def m_negate_crossvar():
    lay = ma.layout(2)
    lead, lag, d = lay["leads"], lay["lags"], lay["d"]
    t, X = gbm.simulate_correlated_gbm(X0, 0.05, SIG, CORR, T, 200, 20, seed=2)
    err = 0.0
    for p in range(X[0].shape[0]):
        s = signature.sig(ma.enlarged(t, [X[0][p], X[1][p]]), 2)
        c = lambda w: s[signature.word_index(w, d)]
        cv_bad = -(c((lead[0], lag[1])) - c((lag[0], lead[1])))   # negated
        cv_direct = np.sum(np.diff(X[0][p]) * np.diff(X[1][p]))
        err = max(err, abs(cv_bad - cv_direct))
    report("negate cross-variation", f"M-I err={err:.1e}", err > 1e-8)


# ---- M-overridge: over-ridge erases the pairs position ----------------------
def m_over_ridge():
    corr = np.array([[1.0, 0.95], [0.95, 1.0]])
    t, X = gbm.simulate_correlated_gbm(X0, 0.0, SIG, corr, T, 50, 60_000, seed=0)
    es, _ = ma.expected_signature(t, X, 2)
    payoff = ma.spread_forward_covector(X0, 0.0, 0, 1, 2)
    basis = ma.strategy_basis(2, 0)
    A, b, idx = ma.mean_variance_system(payoff, 0.0, basis, 2, es)
    ell = scaling.ridged_solve(A, b, lam=10.0)             # over-ridge
    th = [ell[idx[(m, ())]] for m in range(2)]
    report("over-ridge (lam=10)", f"MH4c spread theta=({th[0]:+.2f},{th[1]:+.2f})",
           not (abs(th[0] - 1) < 5e-2 and abs(th[1] + 1) < 5e-2))


# ---- M-theta2-zero: disable cross-hedging -----------------------------------
def m_theta2_zero():
    # force theta^2 = 0 -> hedging X1 with X2 can no longer reduce variance.
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    t, X = gbm.simulate_correlated_gbm(X0, 0.0, SIG, corr, T, 50, 30_000, seed=9)
    p0 = X0[0] - 1.0
    basis = ma.strategy_basis(2, 1)
    ell = [{w: 0.0 for w in basis}, {w: 0.0 for w in basis}]   # theta2 = 0 (and theta1=0)
    L = ma.shortfall(ell, lambda tt, xx: xx[0][:, -1] - 1.0, p0, t, X, 1, 2)
    L0 = X[0][:, -1] - 1.0 - p0
    report("force theta2=0", f"MH5 var {L0.var():.3e} vs {L.var():.3e}",
           abs(L.var(ddof=1) - L0.var(ddof=1)) < 1e-6)


if __name__ == "__main__":
    print("multi-asset mutations (CAUGHT = gate went red, as it must):")
    m_swap_trade_letters()
    m_zero_cross()
    m_negate_crossvar()
    m_over_ridge()
    m_theta2_zero()
    print(f"\n{sum(results)}/{len(results)} mutations caught.")
    sys.exit(0 if all(results) else 1)
