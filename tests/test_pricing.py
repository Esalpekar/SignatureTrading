"""Category C — pricing correctness vs closed form.

C1 (level-1 forward) lives in test_forward.py and is retained from v0. Here:
C2 lifts the pricing check to a path-dependent level-2 claim (arithmetic Asian),
C3 checks linearity of the covector pairing, C4 the empty-word / r=0 edges.
"""
import numpy as np

from sigcore import gbm, price, signature
import helpers as H

X0, R, SIGMA, T, K = 1.0, 0.03, 0.2, 1.0, 1.0
N_STEPS, N_PATHS, D = 250, 100_000, 2


def test_C2_arithmetic_asian_forward():
    closed = np.exp(-R * T) * (H.asian_expected_average(X0, R, T) - K)
    const, vec = H.asian_covector(X0, K, T, D)
    idx = signature.word_index((1, 0), D)

    # analytic arm: feed the oracle level-2 coordinate E[int (X-X0) dt]; exact.
    es_an = np.zeros(idx + 1)
    es_an[idx] = H.asian_level2_target(X0, R, T)
    p_an = price.price(const, vec, es_an, R, T)
    print(f"\n[C2] closed={closed:.6f}  analytic={p_an:.6f}  "
          f"|err|={abs(p_an - closed):.1e}")
    assert abs(p_an - closed) < 1e-6

    # MC arm: price embedding (t, X), level >= 2, within 4*SE.
    times, paths = gbm.simulate(X0, R, SIGMA, T, N_STEPS, N_PATHS, seed=0)
    from sigcore import embed
    mc, se = H.mc_expected_signature(embed.time_augment(times, paths), 2)
    p_mc = price.price(const, vec, mc, R, T)
    band = 4.0 * np.exp(-R * T) * (1.0 / T) * se[idx]
    print(f"     mc={p_mc:.6f}  |err|={abs(p_mc - closed):.1e}  band(4SE)={band:.1e}")
    assert abs(p_mc - closed) <= band


def test_C3_pricing_linearity():
    rng = np.random.default_rng(0)
    es = rng.standard_normal(14)                 # arbitrary expected signature
    c1, v1 = -0.3, rng.standard_normal(14)
    c2, v2 = 1.1, rng.standard_normal(14)
    alpha, beta = 2.0, -0.7

    combined = price.price(alpha * c1 + beta * c2,
                           alpha * v1 + beta * v2, es, R, T)
    separate = (alpha * price.price(c1, v1, es, R, T)
                + beta * price.price(c2, v2, es, R, T))
    assert abs(combined - separate) < 1e-10


def test_C4_pricing_edge_cases():
    idx = signature.word_index((1,), D)

    # (i) r = 0: forward -> X0 - K, Asian -> X0 - K.
    const, vec = price.forward_covector(X0, K, D)
    es_fwd = np.zeros(idx + 1)
    es_fwd[idx] = X0 * (np.exp(0.0 * T) - 1.0)        # = 0
    assert abs(price.price(const, vec, es_fwd, 0.0, T) - (X0 - K)) < 1e-10

    ac, av = H.asian_covector(X0, K, T, D)
    aidx = signature.word_index((1, 0), D)
    es_as = np.zeros(aidx + 1)
    es_as[aidx] = H.asian_level2_target(X0, 0.0, T)    # = 0 via the r=0 limit
    assert abs(price.price(ac, av, es_as, 0.0, T) - (X0 - K)) < 1e-10

    # (ii) K = 0: forward -> X0.
    const0, vec0 = price.forward_covector(X0, 0.0, D)
    es2 = np.zeros(idx + 1)
    es2[idx] = X0 * (np.exp(R * T) - 1.0)
    assert abs(price.price(const0, vec0, es2, R, T) - X0) < 1e-10

    # (iii) pure-constant "bond" payoff f = 1 * empty word -> e^{-rT}.
    bond = price.price(1.0, np.zeros(idx + 1), es2, R, T)
    assert abs(bond - np.exp(-R * T)) < 1e-10
