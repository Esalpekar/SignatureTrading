"""T1 - Signature engine vs Fawcett (time-augmented Brownian driver).

De-flaked: the v0 form used a single seed and a fixed 5e-3 tolerance, but the
worst per-seed max coordinate error is ~7.4e-3 (worst ~4.2*SE) — so the fixed
single-seed gate could fail on an unlucky seed and pass on a lucky one, which
trains rerun-until-green. Here it is a seed sweep with a statistically-justified
band: every stochastic coordinate must agree with Fawcett within 5*SE (real
margin above the observed 4.2), deterministic coordinates exactly. A failure now
means a genuine engine bug, not a draw of the dice.
"""
import numpy as np

from sigcore import gbm, embed
from sigcore import expected_signature as es
import helpers as H

T, N_STEPS, N_PATHS, LEVEL, D, SIGMA = 1.0, 250, 100_000, 3, 2, 1.0
N_SEEDS = 10
SE_BAND = 5.0


def _fawcett():
    g1 = np.array([T, 0.0])
    g2 = np.array([[0.0, 0.0], [0.0, T / 2.0]])
    return es.analytic_gaussian((g1, g2), D, LEVEL)


def test_signature_engine_vs_fawcett_seed_sweep():
    analytic = _fawcett()
    worst_ratio, worst_abs = 0.0, 0.0
    for seed in range(N_SEEDS):
        times, W = gbm.simulate_brownian(SIGMA, T, N_STEPS, N_PATHS, seed=seed)
        mc, se = H.mc_expected_signature(embed.time_augment(times, W), LEVEL)
        err = np.abs(mc - analytic)

        det = se <= 1e-9                       # pure-time words: deterministic
        assert np.max(err[det]) < 1e-9
        stoch = ~det
        ratio = err[stoch] / se[stoch]
        worst_ratio = max(worst_ratio, ratio.max())
        worst_abs = max(worst_abs, err.max())
        assert ratio.max() < SE_BAND, (
            f"seed {seed}: max |err|/SE = {ratio.max():.2f} exceeds {SE_BAND}")

    print(f"\n[T1] {N_SEEDS} seeds: worst |err|/SE={worst_ratio:.2f} (band {SE_BAND}), "
          f"worst max|err|={worst_abs:.2e}")


def test_signature_engine_monte_carlo_plumbing():
    # Exercise the real expected_signature.monte_carlo (the per-path averager
    # the rest of the suite's fast batched helper stands in for) on one seed.
    analytic = _fawcett()
    times, W = gbm.simulate_brownian(SIGMA, T, N_STEPS, N_PATHS, seed=0)
    mc, se = es.monte_carlo(embed.time_augment(times, W), LEVEL)
    stoch = se > 1e-9
    assert np.max(np.abs(mc - analytic)[stoch] / se[stoch]) < SE_BAND
