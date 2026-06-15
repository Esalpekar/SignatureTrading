"""Category B — expected-signature estimation accuracy vs the all-level oracle.

Embedding: time-augmented log-price GBM ``(t, log X_t)``, whose expected
signature is ``exp_(x)(T*xi)`` at every level (validated against
``signature.tensor_exp``). This closes the v0 gap that only checked level 1.

Per-level absolute thresholds were calibrated once over 20 seeds at
n_paths=1e5, n_steps=250 (worst-seed per-level max error was roughly
1.5e-3, 8e-4, 4e-4, 1e-4 for levels 1-4); the gates below sit ~2x above that
so every seed passes (see B5). Numbers, not just pass/fail, are reported.
"""
import numpy as np
import pytest

from sigcore import gbm, signature
import helpers as H

# shared GBM params
X0, R, SIGMA, T = 1.0, 0.03, 0.2, 1.0
N_STEPS, N_PATHS, LEVEL = 250, 100_000, 4

# calibrated per-level absolute-error gates (level 1..4)
B1_THRESH = [3.0e-3, 2.0e-3, 1.0e-3, 5.0e-4]
LEVEL_COUNTS = [2 ** L for L in range(1, LEVEL + 1)]
LEVEL_OFFSETS = np.concatenate([[0], np.cumsum(LEVEL_COUNTS)])[:-1]


def _mc_and_oracle(n_paths, n_steps, seed, level=LEVEL):
    times, paths = gbm.simulate(X0, R, SIGMA, T, n_steps, n_paths, seed=seed)
    mc, se = H.mc_expected_signature(H.log_price_embed(times, paths), level)
    oracle = H.gbm_log_expected_signature(R, SIGMA, T, level)
    return mc, se, oracle


def _per_level_max_err(mc, oracle):
    errs = np.abs(mc - oracle)
    return [errs[o:o + c].max() for o, c in zip(LEVEL_OFFSETS, LEVEL_COUNTS)]


def test_B1_all_level_oracle():
    mc, se, oracle = _mc_and_oracle(N_PATHS, N_STEPS, seed=0)
    per_level = _per_level_max_err(mc, oracle)
    print("\n[B1] per-level max |MC - oracle| (n=1e5, steps=250):")
    for L, (err, thr) in enumerate(zip(per_level, B1_THRESH), start=1):
        print(f"  L{L}: max={err:.2e}  threshold={thr:.1e}")
        assert err < thr, f"level {L} max error {err:.2e} exceeds {thr:.1e}"

    # secondary diagnostic: fraction of stochastic coords within 4*SE
    stoch = se > 1e-9
    within4 = float(np.mean(np.abs(mc - oracle)[stoch] <= 4 * se[stoch]))
    print(f"  within-4SE (stochastic coords): {within4 * 100:.0f}%")
    assert within4 >= 0.90


def test_B2_per_level_error_table():
    mc, se, oracle = _mc_and_oracle(N_PATHS, N_STEPS, seed=0)
    errs = np.abs(mc - oracle)
    print("\n[B2] level | n_coords | max|err| | mean|err| | mean SE")
    for L, (o, c) in enumerate(zip(LEVEL_OFFSETS, LEVEL_COUNTS), start=1):
        blk, sblk = errs[o:o + c], se[o:o + c]
        print(f"  L{L} | {c:2d} | {blk.max():.2e} | {blk.mean():.2e} | "
              f"{sblk.mean():.2e}")
    # error climbs nowhere catastrophically; basic ordering sanity
    assert errs.max() < max(B1_THRESH)


def test_B3_sampling_error_scaling():
    # SE ~ 1/sqrt(n_paths): SE(4n)/SE(n) ~ 0.5 within +-15%, per level.
    _, se_n, _ = _mc_and_oracle(25_000, N_STEPS, seed=0)
    _, se_4n, _ = _mc_and_oracle(100_000, N_STEPS, seed=0)
    print("\n[B3] SE(4n)/SE(n) per representative coord (target 0.5):")
    for L, (o, c) in enumerate(zip(LEVEL_OFFSETS, LEVEL_COUNTS), start=1):
        idx = o + c - 1                                   # last coord of level
        ratio = se_4n[idx] / se_n[idx]
        print(f"  L{L} idx{idx}: ratio={ratio:.3f}")
        assert 0.5 * 0.85 <= ratio <= 0.5 * 1.15


def test_B4_discretisation_bias():
    # Common-random-number subsampling isolates bias from sampling noise: build
    # coarse grids by aggregating the finest increments (same paths). Use a
    # larger sigma so bias rises above the sampling floor and is visible.
    sigma = 0.4
    n_paths, fine = 100_000, 500
    level = 3
    oracle = H.gbm_log_expected_signature(R, sigma, T, level)
    rng = np.random.default_rng(0)
    dt_f = T / fine
    incr = ((R - 0.5 * sigma ** 2) * dt_f
            + sigma * np.sqrt(dt_f) * rng.standard_normal((n_paths, fine)))

    grids = [10, 25, 50, 125, 250, 500]
    bias = []
    for ns in grids:
        agg = incr.reshape(n_paths, ns, fine // ns).sum(axis=2)
        logp = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(agg, axis=1)], axis=1)
        from sigcore import embed
        emb = embed.time_augment(np.linspace(0.0, T, ns + 1), logp)
        mc, _ = H.mc_expected_signature(emb, level)
        bias.append(float(np.linalg.norm((mc - oracle)[2:])))  # levels 2..3

    print("\n[B4] ||err lvl>=2|| vs n_steps (CRN, sigma=0.4):")
    for ns, b in zip(grids, bias):
        print(f"  n_steps={ns:4d}  bias={b:.3e}")
    # monotone non-increasing (allow 5% wobble at the sampling floor)
    for a, b in zip(bias, bias[1:]):
        assert b <= a * 1.05
    assert bias[0] > 2 * bias[-1]      # coarse grid clearly worse than fine


def test_B5_seed_robustness():
    # Every one of >=20 seeds must pass the B1 per-level thresholds.
    seeds = range(20)
    worst = np.zeros(LEVEL)
    fails = []
    for sd in seeds:
        mc, _, oracle = _mc_and_oracle(N_PATHS, N_STEPS, seed=sd)
        per_level = _per_level_max_err(mc, oracle)
        worst = np.maximum(worst, per_level)
        if any(e >= t for e, t in zip(per_level, B1_THRESH)):
            fails.append((sd, per_level))
    print("\n[B5] worst per-level max error over 20 seeds:")
    for L in range(LEVEL):
        print(f"  L{L + 1}: worst={worst[L]:.2e}  threshold={B1_THRESH[L]:.1e}")
    assert not fails, f"seeds failing B1 thresholds: {fails}"
