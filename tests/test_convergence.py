"""T3 - Monte-Carlo expected-signature convergence (slope ~ -0.5)."""
import numpy as np

from sigcore import gbm, embed
from sigcore import expected_signature as es


def test_mc_convergence_slope():
    T = 1.0
    n_steps = 250
    level = 3
    d = 2
    sigma = 1.0

    g1 = np.array([T, 0.0])
    g2 = np.array([[0.0, 0.0], [0.0, T / 2.0]])
    analytic = es.analytic_gaussian((g1, g2), d, level)

    ns = [100, 1_000, 10_000, 100_000]
    errs = []
    for n in ns:
        times, W = gbm.simulate_brownian(sigma, T, n_steps, n, seed=0)
        mc, _ = es.monte_carlo(embed.time_augment(times, W), level)
        errs.append(np.linalg.norm(mc - analytic))

    slope = np.polyfit(np.log(ns), np.log(errs), 1)[0]
    assert -0.6 <= slope <= -0.4, f"log-log slope {slope:.2f} outside [-0.6, -0.4]"
