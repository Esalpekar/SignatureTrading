"""U — Universality / truncation convergence.

Every payoff priced so far (forward, Asian) IS exactly a signature coordinate,
so its truncation error is identically zero. That never exercises the method's
central claim: a linear functional of the *truncated* signature approximates an
arbitrary continuous payoff, with error -> 0 as the level N grows.

Test payoff: the power claim ``X_T^p`` with non-integer ``p`` (closed-form
lognormal price). Its covector is built analytically at each level N from the
Taylor expansion of ``x -> x^p`` about ``X0``: ``(X_T - X0)^k`` is the shuffle
power of the price-letter, so ``f = sum_{k<=N} c_k * (price-letter)^{shuffle k}``.
Pricing is the genuine ``<f, E^Q[S]>`` through ``price.price`` and
``word_index`` (no regression — this isolates truncation from any fitting).
"""
from math import factorial

import numpy as np
import pytest
import iisignature

from sigcore import signature, price, gbm, embed
import helpers as H

X0, R, SIGMA, T, D = 1.0, 0.03, 0.2, 1.0, 2
NMAX = 5
TOL = 1e-4                      # both p reach < 1e-4 by N=5 (see calibration)
PRICE_LETTER = (1,)


def _sig_len(level):
    return sum(D ** k for k in range(1, level + 1))


def _analytic_expected_signature(level):
    """E^Q[S] populated where the power covector reads it: the words
    ``(1,)*k`` carry ``E[(X_T - X0)^k] / k!``. Other coords are unused (the
    covector is zero there), so leaving them zero is exact, not an approximation."""
    es = np.zeros(_sig_len(level))
    for k in range(1, level + 1):
        es[signature.word_index(PRICE_LETTER * k, D)] = (
            H.central_moment(X0, R, SIGMA, T, k) / factorial(k))
    return es


def _power_covector(p, n, level):
    """(const, vec) for the level-n truncation of ``x^p`` about X0."""
    const = H.taylor_coeff(X0, p, 0)               # c_0 = X0^p (empty word)
    vec = np.zeros(_sig_len(level))
    for k in range(1, n + 1):
        ck = H.taylor_coeff(X0, p, k)
        for word, mult in H.shuffle_power(PRICE_LETTER, k).items():
            vec[signature.word_index(word, D)] += ck * mult
    return const, vec


@pytest.mark.parametrize("p", [1.5, 0.5])
def test_U_power_claim_truncation_convergence(p):
    closed = H.power_claim_price(X0, R, SIGMA, T, p)
    es = _analytic_expected_signature(NMAX)

    errs = []
    for n in range(1, NMAX + 1):
        const, vec = _power_covector(p, n, NMAX)
        errs.append(abs(price.price(const, vec, es, R, T) - closed))

    print(f"\n[U] X_T^{p}  closed={closed:.6f}")
    for n, e in enumerate(errs, start=1):
        print(f"  N={n}: |err|={e:.2e}")

    # error decreases monotonically with the truncation level ...
    for a, b in zip(errs, errs[1:]):
        assert b < a, f"non-monotone truncation error: {errs}"
    # ... and reaches the stated tolerance by N=5.
    assert errs[-1] < TOL


def test_U_path_dependent_payoff_converges_to_mc():
    # Stretch: a NON-polynomial functional of the *trajectory* — exp of the
    # running average A = (1/T) int X dt. Its covector is the Taylor series of
    # exp about A0 = X0, where (A - X0) = (1/T) * <e_(1,0), S> is the mixed
    # time-price word (1,0). So f populates the (1,0) shuffle-power coordinates
    # — the genuinely path-dependent, mixed-channel slice that X_T^p never
    # touches. No closed form: validate against the direct Monte-Carlo price.
    #
    # Both <f_N, E[S]> and the direct price are evaluated on the SAME paths, so
    # the MC sampling error cancels in their difference and what remains is pure
    # truncation error -> 0 as N grows.
    x0, r, sigma, T = 1.0, 0.03, 0.2, 1.0
    nmax = 4
    mixed = (1, 0)                                   # "price then time"
    times, paths = gbm.simulate(x0, r, sigma, T, 100, 20_000, seed=1)
    sigs = iisignature.sig(np.ascontiguousarray(embed.time_augment(times, paths)),
                           2 * nmax)                 # (1,0)^{shuffle k} has length 2k
    es_mc = sigs.mean(axis=0)

    i10 = signature.word_index(mixed, D)
    avg = x0 + (1.0 / T) * sigs[:, i10]              # per-path running average
    direct = np.exp(-r * T) * np.mean(np.exp(avg))   # direct MC price of exp(A)

    errs = []
    for n in range(1, nmax + 1):
        const = np.exp(x0)                           # c_0 = exp(A0), A0 = x0
        vec = np.zeros(es_mc.shape[0])
        for k in range(1, n + 1):
            ck = np.exp(x0) / factorial(k) * (1.0 / T) ** k
            for word, mult in H.shuffle_power(mixed, k).items():
                vec[signature.word_index(word, D)] += ck * mult
        errs.append(abs(price.price(const, vec, es_mc, r, T) - direct))

    print(f"\n[U-stretch] exp(running average)  direct MC={direct:.6f}")
    for n, e in enumerate(errs, start=1):
        print(f"  N={n}: |<f,E[S]> - MC|={e:.2e}")
    for a, b in zip(errs, errs[1:]):
        assert b < a, f"non-monotone path-dependent convergence: {errs}"
    assert errs[-1] < 1e-4


def test_U_polynomial_payoff_is_exact_at_its_degree():
    # Sanity contrast: an integer-power (polynomial) payoff is *exactly*
    # representable, so its error hits ~0 at N = degree (nothing to approximate).
    p = 2.0
    closed = H.power_claim_price(X0, R, SIGMA, T, p)
    es = _analytic_expected_signature(NMAX)
    const, vec = _power_covector(p, 2, NMAX)
    assert abs(price.price(const, vec, es, R, T) - closed) < 1e-10
