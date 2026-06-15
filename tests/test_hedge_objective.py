"""H0 — Objective cross-check (BEDROCK).

The shuffle-assembled objective must equal a direct Monte-Carlo estimate of
E[P(L)] computed by literally simulating, applying theta_t = <ell, S_{0,t}>,
forming L, and averaging P(L) — code that shares nothing with the covector
algebra. Two arms:

  * per-path EXACT identities (machine precision, no MC noise): they pin the
    trade-letter append, the loss covector, and the shuffle powers separately;
  * the spec's independent-sample agreement within 4*SE.

If H0 fails, nothing downstream is meaningful.
"""
import numpy as np
import iisignature
import pytest

from sigcore import gbm
from sigcore.hedge import embedding as emb, objective as obj, strategy, pnl
import helpers as H

X0, R, SIGMA, T, K = 1.0, 0.05, 0.3, 1.0, 1.0
GAMMA, DELTA = 0.5, 0.2                       # delta > 3 gamma^2/8 = 0.094: convex
MV = {2: 1.0}
QUARTIC = {2: 1.0, 3: GAMMA, 4: DELTA}


def _p0(kind):
    if kind == "forward":
        return H.forward_closed_form(X0, K, R, T)
    return np.exp(-R * T) * (H.asian_expected_average(X0, R, T) - K)


def _payoff_cov(kind):
    return emb.forward_covector(X0, K) if kind == "forward" else emb.asian_covector(X0, K, T)


def _payoff_fn(kind):
    return pnl.forward_payoff if kind == "forward" else pnl.asian_payoff


def _random_ell(basis, seed):
    rng = np.random.default_rng(seed)
    return {w: float(v) for w, v in zip(basis, rng.standard_normal(len(basis)))}


# (kind, depth, penalty, label) — levels stay <= 8
CASES = [
    ("forward", 1, MV, "mv"),
    ("asian", 1, MV, "mv"),
    ("forward", 0, QUARTIC, "quartic"),
    ("forward", 1, QUARTIC, "quartic"),
    ("asian", 1, QUARTIC, "quartic"),
]


@pytest.mark.parametrize("kind,depth,penalty,label", CASES)
def test_H0_per_path_identities(kind, depth, penalty, label):
    basis = obj.strategy_basis(depth)
    ell = _random_ell(basis, seed=hash((kind, depth, label)) % 1000)
    p0 = _p0(kind)
    payoff_cov = _payoff_cov(kind)
    loss = obj.loss_covector(payoff_cov, p0, ell)
    level = obj.required_signature_level(loss, max(penalty))

    times, paths = gbm.simulate(X0, R, SIGMA, T, 30, 300, seed=7)
    sigs = iisignature.sig(np.ascontiguousarray(emb.enlarged_batch(times, paths)), level)

    payoff = _payoff_fn(kind)(times, paths, K)
    trade = obj.cov.append_letter(ell, emb.TRADE_LETTER)
    loss_pow = {k: obj.cov.shuffle_power(loss, k) for k in penalty}

    max_err_pnl = max_err_loss = max_err_pen = 0.0
    for i in range(paths.shape[0]):
        s = sigs[i]
        # (1) direct feedback P&L == <ell.a, S>
        theta = strategy.positions(ell, times, paths[i], level)
        pnl_direct = np.sum(theta[:-1] * np.diff(paths[i]))
        pnl_cov = obj.cov.contract(trade, s, emb.N_CHANNELS)
        max_err_pnl = max(max_err_pnl, abs(pnl_direct - pnl_cov))
        # (2) realised shortfall == <L, S>
        L_direct = payoff[i] - p0 - pnl_direct
        L_cov = obj.cov.contract(loss, s, emb.N_CHANNELS)
        max_err_loss = max(max_err_loss, abs(L_direct - L_cov))
        # (3) <P^sh(L), S> == P(L)
        P_cov = sum(penalty[k] * obj.cov.contract(loss_pow[k], s, emb.N_CHANNELS)
                    for k in penalty)
        P_direct = sum(penalty[k] * L_direct ** k for k in penalty)
        max_err_pen = max(max_err_pen, abs(P_cov - P_direct))

    assert max_err_pnl < 1e-8, f"P&L append wiring: {max_err_pnl:.1e}"
    assert max_err_loss < 1e-8, f"loss covector: {max_err_loss:.1e}"
    assert max_err_pen < 1e-7, f"shuffle powers: {max_err_pen:.1e}"


@pytest.mark.parametrize("kind,depth,penalty,label", CASES)
def test_H0_objective_vs_direct_mc(kind, depth, penalty, label):
    basis = obj.strategy_basis(depth)
    ell = _random_ell(basis, seed=(hash((kind, depth, label)) % 1000) + 1)
    p0 = _p0(kind)
    payoff_cov = _payoff_cov(kind)
    loss = obj.loss_covector(payoff_cov, p0, ell)
    level = obj.required_signature_level(loss, max(penalty))

    # E[S] from one seed; direct-MC shortfall from an independent seed.
    # Scale paths down for deep levels (sig array ~ n_paths * 3^level) — the
    # 4*SE band simply widens, so the cross-check stays valid.
    n_paths = {4: 60_000}.get(level, 8_000)
    t_es, p_es = gbm.simulate(X0, R, SIGMA, T, 40, n_paths, seed=11)
    es, _ = emb.expected_signature(t_es, p_es, level)
    R_shuffle = obj.objective_value(loss, penalty, es)

    t_d, p_d = gbm.simulate(X0, R, SIGMA, T, 40, n_paths, seed=22)
    L = pnl.shortfall(ell, _payoff_fn(kind), p0, t_d, p_d, level, K)
    PL = sum(penalty[k] * L ** k for k in penalty)
    direct, se = PL.mean(), PL.std(ddof=1) / np.sqrt(PL.size)

    # R_shuffle is itself a sample mean of P(L) over the (equal-size, independent)
    # E[S] paths, so the difference of the two means has combined SE ~ sqrt(2)*se.
    band = 4 * np.sqrt(2.0) * se
    print(f"\n[H0:{kind}/{label}/d{depth}] shuffle={R_shuffle:.5f}  "
          f"direct={direct:.5f}  |err|={abs(R_shuffle-direct):.2e}  band={band:.2e}")
    assert abs(R_shuffle - direct) <= band
