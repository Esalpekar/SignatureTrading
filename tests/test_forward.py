"""T4 - Forward price vs closed form X0 - K*e^{-rT}."""
import numpy as np

from sigcore import gbm, embed, signature, price
from sigcore import expected_signature as es


def test_forward_analytic_and_mc():
    x0 = 1.0
    r = 0.03
    sigma = 0.2
    T = 1.0
    K = 1.0
    n_steps = 250
    n_paths = 100_000
    level = 3
    d = 2

    closed_form = x0 - K * np.exp(-r * T)
    const, vec = price.forward_covector(x0, K, d)
    price_idx = signature.word_index((1,), d)

    # --- analytic pipeline: exact to floating point (validates the wiring) ---
    es_analytic = np.zeros(price_idx + 1)
    es_analytic[price_idx] = x0 * (np.exp(r * T) - 1.0)     # E[X_T - X_0]
    p_analytic = price.price(const, vec, es_analytic, r, T)
    assert abs(p_analytic - closed_form) < 1e-10

    # --- monte-carlo pipeline: within the reported MC band (~2*SE) ----------
    times, paths = gbm.simulate(x0, r, sigma, T, n_steps, n_paths, seed=0)
    mc, se = es.monte_carlo(embed.time_augment(times, paths), level)
    p_mc = price.price(const, vec, mc, r, T)
    band = 2.0 * np.exp(-r * T) * se[price_idx]
    assert abs(p_mc - closed_form) <= band, (
        f"MC price err {abs(p_mc - closed_form):.2e} exceeds band {band:.2e}"
    )
