"""M0 + M-I — the non-negotiable FIRST gates for multi-asset.

M0 validates the JOINT expected signature against the closed-form tensor
exponential (cross coordinate = rho_ij T, antisymmetric Levy-area part = 0).
M-I validates the new cross-asset Ito mechanism (cross integral and cross
variation) from the joint lead-lag. No hedging code may be trusted until these
are green — no single-asset test can vet a cross-asset coordinate.
"""
import numpy as np
import iisignature
import pytest

from sigcore import gbm, signature
from sigcore.hedge import multiasset as ma
import helpers as H

T = 1.0
CORR2 = np.array([[1.0, 0.5], [0.5, 1.0]])
CORR3 = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]])


def _joint_mc(times, series_list, level):
    emb = H.time_augment_multi(times, *series_list)
    sigs = iisignature.sig(np.ascontiguousarray(emb), level)
    return sigs.mean(axis=0), sigs.std(axis=0, ddof=1) / np.sqrt(sigs.shape[0])


@pytest.mark.parametrize("corr", [CORR2, CORR3])
def test_M0_engine_oracle_brownian(corr):
    M = corr.shape[0]
    d = 1 + M
    level = 3
    times, W = gbm.simulate_correlated_brownian_multi(corr, T, 200, 100_000, seed=0)
    mc, se = _joint_mc(times, W, level)
    oracle = H.correlated_oracle(corr, T, level)

    stoch = se > 1e-9
    worst = float((np.abs(mc - oracle)[stoch] / se[stoch]).max())
    print(f"\n[M0 BM M={M}] all-level worst |err|/SE = {worst:.2f}")
    assert worst < 5.0
    for i in range(M):
        for j in range(i + 1, M):
            ci, cj = 1 + i, 1 + j
            sym = mc[signature.word_index((ci, cj), d)] + mc[signature.word_index((cj, ci), d)]
            anti = mc[signature.word_index((ci, cj), d)] - mc[signature.word_index((cj, ci), d)]
            se_sym = np.sqrt(se[signature.word_index((ci, cj), d)] ** 2
                             + se[signature.word_index((cj, ci), d)] ** 2)
            print(f"  pair ({i},{j}): sym={sym:.4f} target={corr[i, j] * T:.4f}  anti={anti:.4f}")
            assert abs(sym - corr[i, j] * T) <= 4 * se_sym
            assert abs(anti) <= 4 * se_sym


def test_M0_engine_oracle_log_gbm():
    M = 2
    d = 1 + M
    level = 3
    sig = np.array([0.3, 0.25])
    r = 0.03
    times, X = gbm.simulate_correlated_gbm([1.0, 1.0], r, sig, CORR2, T, 200, 100_000, seed=1)
    logX = [np.log(x) for x in X]
    mc, se = _joint_mc(times, logX, level)
    oracle = H.correlated_oracle(CORR2, T, level, sigmas=sig, r=r)
    stoch = se > 1e-9
    worst = float((np.abs(mc - oracle)[stoch] / se[stoch]).max())
    sym = mc[signature.word_index((1, 2), d)] + mc[signature.word_index((2, 1), d)]
    target = sig[0] * sig[1] * CORR2[0, 1] * T
    print(f"\n[M0 logGBM] worst |err|/SE={worst:.2f}  cross sym={sym:.5f} target={target:.5f}")
    assert worst < 5.0
    se_sym = np.sqrt(se[signature.word_index((1, 2), d)] ** 2
                     + se[signature.word_index((2, 1), d)] ** 2)
    assert abs(sym - target) <= 4 * se_sym


def test_MI_cross_asset_ito_recovery():
    # From the joint lead-lag: int X^i dX^j (Ito) and the cross variation
    # [X^i, X^j], recovered per-path and matched to the literal sums.
    M = 2
    lay = ma.layout(M)
    d = lay["d"]
    lead, lag = lay["leads"], lay["lags"]
    times, X = gbm.simulate_correlated_gbm([1.0, 1.0], 0.05, [0.3, 0.25], CORR2,
                                           T, 200, 25, seed=2)
    i, j = 0, 1
    max_ito = max_cv = max_strat = 0.0
    for p in range(X[0].shape[0]):
        Xi, Xj = X[i][p], X[j][p]
        s = signature.sig(ma.enlarged(times, [Xi, Xj]), 2)
        c = lambda w: s[signature.word_index(w, d)]
        # Ito cross integral: <(lag_i, lead_j), S> + X^i_0 * dX^j
        ito = c((lag[i], lead[j])) + Xi[0] * (Xj[-1] - Xj[0])
        ito_direct = np.sum(Xi[:-1] * np.diff(Xj))                 # literal left-point
        max_ito = max(max_ito, abs(ito - ito_direct))
        # cross variation: <(lead_i, lag_j)> - <(lag_i, lead_j)>
        cv = c((lead[i], lag[j])) - c((lag[i], lead[j]))
        cv_direct = np.sum(np.diff(Xi) * np.diff(Xj))
        max_cv = max(max_cv, abs(cv - cv_direct))
        # Ito = Stratonovich - 1/2 [X^i, X^j]
        strat = np.sum(0.5 * (Xi[:-1] + Xi[1:]) * np.diff(Xj))
        max_strat = max(max_strat, abs(ito_direct - (strat - 0.5 * cv_direct)))
    print(f"\n[M-I] max|err| ito={max_ito:.1e}  cross-var={max_cv:.1e}  "
          f"Ito-Strat identity={max_strat:.1e}")
    assert max_ito < 1e-8 and max_cv < 1e-8 and max_strat < 1e-8
