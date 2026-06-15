"""I — Itô recovery with a state-dependent integrand.

T2/D3 only check the lead-lag *area* = 1/2 QV, i.e. the Itô integral of a
CONSTANT integrand. Hedging P&L is ``int theta_t dX_t`` with theta varying along
the path. This test recovers ``int_0^T X_t dX_t`` (a state-dependent integrand)
from the lead-lag signature and checks it against the exact discrete Itô value
``1/2 (X_T^2 - X_0^2) - 1/2 QV`` per path. It is an exact identity of the
discrete lead-lag construction, so the residual is float noise (< 1e-8).
"""
import numpy as np

from sigcore import gbm, embed, signature
import helpers as H


def _ito_price_against_price(series):
    """Recover ``int X dX`` (left-point Itô) from the lead-lag signature.

    The level-2 coordinate ``<lag, lead>`` (word (1,0): integrate the lagging
    price against the leading price) equals ``sum_k (X_k - X_0) dX_k`` because
    the signature is translation invariant; adding ``X_0 * (X_T - X_0)`` (the
    level-1 coordinate scaled by X_0) restores ``sum_k X_k dX_k``. The word
    index is *located* via word_index, not hard-coded.
    """
    s = signature.sig(embed.lead_lag(series), 2)
    i_lag_lead = signature.word_index((1, 0), 2)
    increment = series[-1] - series[0]               # level-1 price coordinate
    return s[i_lag_lead] + series[0] * increment


def test_I_state_dependent_ito_recovery():
    times, paths = gbm.simulate(1.0, 0.05, 0.3, 1.0, 300, 25, seed=3)
    max_err = 0.0
    for series in paths:
        recovered = _ito_price_against_price(series)
        qv = np.sum(np.diff(series) ** 2)
        exact = 0.5 * (series[-1] ** 2 - series[0] ** 2) - 0.5 * qv
        max_err = max(max_err, abs(recovered - exact))
    print(f"\n[I] max per-path |recovered - exact int X dX| = {max_err:.2e}")
    assert max_err < 1e-8


def test_I_distinguishes_from_constant_integrand():
    # Guard: the state-dependent integral is genuinely different from 1/2 QV
    # (the constant-integrand quantity T2/D3 already cover), so this test adds
    # real coverage rather than restating the area identity.
    times, paths = gbm.simulate(1.0, 0.05, 0.3, 1.0, 300, 5, seed=1)
    for series in paths:
        ito = _ito_price_against_price(series)
        half_qv = 0.5 * np.sum(np.diff(series) ** 2)
        assert abs(ito - half_qv) > 1e-3
