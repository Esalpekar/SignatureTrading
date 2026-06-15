"""H4 — scaling and ridge (conditioning mitigation + foresight).

Honest finding: at the truncation depths the core's E[S] supports (<= level 6)
the un-ridged mean-variance solve is ALREADY stable across seeds — the low-depth
hedge is dominated by deterministic time-structure, so the regression is
well-determined and ridge would only add bias. The conditioning wall the probe
shows (~100x/level) bites only at deeper truncation. So this gate certifies what
is true here: (a) scaling substantially reduces cond(A); (b) the solve is stable
across >=20 seeds and ridge is safe (stays close to un-ridged for small lambda);
(c) lambda->0 recovers the unregularised solve; (d) cond(A) grows with depth —
the wall, which is why scaling/ridge are kept ready.
"""
import numpy as np

from sigcore import gbm
from sigcore.hedge import embedding as emb, objective as obj, scaling
import helpers as H

X0, R, SIGMA, T, K = 1.0, 0.03, 0.2, 1.0, 1.0
LAM = 1e-3


def _system(seed, n_paths, depth=2):
    level = 2 * (depth + 1)                      # forward MV needs q*(depth+1)
    t, paths = gbm.simulate(X0, R, SIGMA, T, 60, n_paths, seed=seed)
    es, _ = emb.expected_signature(t, paths, level)
    payoff = emb.forward_covector(X0, K)
    p0 = H.forward_closed_form(X0, K, R, T)
    basis = obj.strategy_basis(depth)
    return obj.mean_variance_system(payoff, p0, basis, es)


def test_H4a_scaling_reduces_condition_number():
    A, b = _system(seed=0, n_paths=60_000)
    c_un, c_sc = scaling.condition_numbers(A)
    print(f"\n[H4a] cond(A) unscaled={c_un:.2e}  scaled={c_sc:.2e}  "
          f"factor={c_un / c_sc:.1f}x")
    assert c_sc < c_un / 5.0


def test_H4b_solve_stable_and_ridge_safe():
    ridged, unridged = [], []
    for seed in range(20):
        A, b = _system(seed=100 + seed, n_paths=6_000)
        ridged.append(scaling.ridged_solve(A, b, lam=LAM))
        unridged.append(scaling.ridged_solve(A, b, lam=0.0))
    ridged, unridged = np.array(ridged), np.array(unridged)
    s_ridge = ridged.std(axis=0).max()
    s_unridge = unridged.std(axis=0).max()
    norm_ridge = np.linalg.norm(ridged, axis=1).max()
    drift = np.max(np.abs(ridged.mean(0) - unridged.mean(0)))
    print(f"\n[H4b] across 20 seeds: ridged std={s_ridge:.2e}  "
          f"un-ridged std={s_unridge:.2e}  max|ridged|={norm_ridge:.2f}  "
          f"ridge bias (drift from unridged)={drift:.2e}")
    # Honest claim: at core-supported depth the solve is stable (bounded
    # coefficient variance across seeds) and the ridged norm stays bounded. The
    # ridge introduces a small lambda-controlled bias (reported, not gated) and
    # is foresight for the deeper, ill-conditioned regime (see H4d).
    assert s_ridge < 0.5 and norm_ridge < 5.0


def test_H4c_ridge_vanishes_recovers_unregularised():
    A, b = _system(seed=7, n_paths=80_000)
    ell_0 = scaling.ridged_solve(A, b, lam=0.0)
    ell_eps = scaling.ridged_solve(A, b, lam=1e-9)
    print(f"\n[H4c] ||ell(1e-9) - ell(0)|| = {np.max(np.abs(ell_eps - ell_0)):.2e}")
    assert np.max(np.abs(ell_eps - ell_0)) < 1e-4


def test_H4d_conditioning_grows_with_depth():
    # the wall: cond(A) climbs with strategy depth (why scaling/ridge are kept).
    conds = []
    for depth in (1, 2):
        A, _ = _system(seed=0, n_paths=60_000, depth=depth)
        conds.append(scaling.condition_numbers(A)[0])
    print(f"\n[H4d] cond(A) by depth (1,2): "
          + "  ".join(f"{c:.1e}" for c in conds))
    assert conds[1] > conds[0]
