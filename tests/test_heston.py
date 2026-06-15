"""Category D — Heston (model-free identities + partial closed forms).

Base parameters satisfy Feller (2*kappa*theta >= xi^2) for D1-D4; D5 uses a
Feller-violating set to exercise the variance clamp.
"""
import numpy as np
import iisignature

from sigcore import heston, signature, embed
import helpers as H

# Feller-satisfying base: 2*kappa*theta = 0.16 >= xi^2 = 0.09
X0, V0, R = 1.0, 0.04, 0.03
KAPPA, THETA, XI, RHO = 2.0, 0.04, 0.3, -0.7
T, N_STEPS, N_PATHS = 1.0, 250, 100_000


def _sim(n_paths=N_PATHS, n_steps=N_STEPS, seed=0, **kw):
    p = dict(x0=X0, v0=V0, r=R, kappa=KAPPA, theta=THETA, xi=XI, rho=RHO,
             T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    p.update(kw)
    return heston.simulate(**p)


def test_D1_identities_on_heston_paths():
    # Shuffle (A2) and Chen (A3) must hold per-path on stochastic-vol paths.
    times, price_paths, _ = _sim(n_paths=5, n_steps=120, seed=0)
    level = 4
    for series in price_paths:
        path = embed.time_augment(times, series)
        s = signature.sig(path, level)

        # shuffle: a couple of representative words
        from sigcore import shuffle
        for u, v in [((0,), (1,)), ((1,), (1, 1)), ((0, 1), (1,))]:
            lhs = H.coeff(s, u, 2) * H.coeff(s, v, 2)
            rhs = sum(m * H.coeff(s, w, 2) for w, m in shuffle.shuffle(u, v).items())
            assert abs(lhs - rhs) < 1e-10

        # Chen
        m = len(path) // 2
        prod = H.tensor_mul_flat(signature.sig(path[:m + 1], level),
                                 signature.sig(path[m:], level), 2, level)
        assert np.max(np.abs(prod - s)) < 1e-10


def test_D2_martingale_level1():
    # Under Q, E[X_T] = X0 * e^{rT} regardless of the vol process.
    _, price_paths, _ = _sim(seed=0)
    xt = price_paths[:, -1]
    mean, se = xt.mean(), xt.std(ddof=1) / np.sqrt(len(xt))
    target = X0 * np.exp(R * T)
    print(f"\n[D2] E[X_T]={mean:.5f} target={target:.5f} "
          f"|err|={abs(mean - target):.2e} 4SE={4 * se:.2e}")
    assert abs(mean - target) <= 4 * se


def test_D3_integrated_variance_via_leadlag():
    # Mean lead-lag area of log-price = 1/2 * realised QV(logX) -> 1/2 E[int v dt].
    _, price_paths, _ = _sim(seed=0)
    log_p = np.log(price_paths)
    area = 0.5 * np.sum(np.diff(log_p, axis=1) ** 2, axis=1)   # = lead-lag area
    mean, se = area.mean(), area.std(ddof=1) / np.sqrt(len(area))
    target = 0.5 * H.heston_integrated_variance(V0, KAPPA, THETA, T)
    print(f"\n[D3] mean area={mean:.6f} target={target:.6f} "
          f"|err|={abs(mean - target):.2e} 4SE={4 * se:.2e}")
    assert abs(mean - target) <= 4 * se


def test_D3_leadlag_matches_quadratic_variation_per_path():
    # Sanity: the lead-lag level-2 antisymmetric coord equals 1/2 QV exactly,
    # so the closed-form check above genuinely exercises the lead-lag embedding.
    times, price_paths, _ = _sim(n_paths=10, n_steps=120, seed=1)
    i01 = signature.word_index((0, 1), 2)
    i10 = signature.word_index((1, 0), 2)
    for series in np.log(price_paths):
        s = signature.sig(embed.lead_lag(series), 2)
        area = 0.5 * (s[i01] - s[i10])
        qv = np.sum(np.diff(series) ** 2)
        assert abs(area - 0.5 * qv) < 1e-9


def test_D4_higher_order_variance_control():
    # Time-augmented log-price, level 3. Half-sample agreement within combined
    # SE, and level-3 relative SE < 10% on coordinates carrying real signal
    # (a magnitude floor excludes near-zero coords where relative SE is
    # ill-conditioned; those are reported, not gated).
    level = 3
    times, price_paths, _ = _sim(seed=0)
    emb = H.log_price_embed(times, price_paths)
    sigs = iisignature.sig(np.ascontiguousarray(emb), level)
    mean = sigs.mean(axis=0)
    se = sigs.std(axis=0, ddof=1) / np.sqrt(sigs.shape[0])

    # --- half-sample agreement within 4*combined SE ---
    h = sigs.shape[0] // 2
    m1, m2 = sigs[:h].mean(0), sigs[h:].mean(0)
    comb = np.sqrt(sigs[:h].std(0, ddof=1) ** 2 + sigs[h:].std(0, ddof=1) ** 2) / np.sqrt(h)
    stoch = comb > 1e-12
    assert np.all(np.abs(m1 - m2)[stoch] <= 4 * comb[stoch])

    # --- level-3 relative SE ---
    blk = slice(6, 14)                       # the 8 level-3 coords
    m, s = mean[blk], se[blk]
    floor = 0.01 * np.linalg.norm(m)
    rel = np.where(np.abs(m) > 1e-14, s / np.abs(m), 0.0)
    above = np.abs(m) > floor
    worst_idx = np.argmax(np.where(above, rel, -1))
    words3 = H.words_of_length(3, 2)
    print(f"\n[D4] worst level-3 relSE (all): {100 * rel.max():.1f}%   "
          f"(above floor): {100 * rel[above].max():.1f}% "
          f"at {words3[worst_idx]}")
    assert rel[above].max() < 0.10


def test_D5_feller_violation_positive_variance():
    # Feller-violating: 2*kappa*theta = 0.08 < xi^2 = 0.25. The clamp must keep
    # variance non-negative on every path and step.
    _, _, var = _sim(n_paths=5_000, n_steps=N_STEPS, seed=0,
                     kappa=1.0, theta=0.04, xi=0.5)
    assert np.all(np.isfinite(var))
    assert var.min() >= 0.0
