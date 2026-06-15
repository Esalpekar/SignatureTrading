"""M — Multi-asset / cross-asset coordinates (brought forward).

The signature *engine* on multi-channel paths — how the joint signature mixes
channels, and the ordering/signs of cross-asset mixed-letter coordinates — is
code that already runs on every path and that all multi-asset work will sit on.
Only the portfolio *allocation* logic is genuinely deferred; this validates the
engine now, with zero portfolio code.

Oracle: two correlated unit Brownian motions ``(W1, W2)``, corr ``rho``,
time-augmented to ``(t, W1, W2)``. The joint expected signature is
``exp_(x)(T*xi)`` with diffusion block ``[[1,rho],[rho,1]]``; in particular the
symmetric cross coordinate ``E[S_(1,2) + S_(2,1)] = rho*T``.
"""
import numpy as np
import iisignature

from sigcore import gbm, signature, shuffle
import helpers as H

RHO, T, N = 0.5, 1.0, 4
N_STEPS, N_PATHS, D = 250, 100_000, 3
SE_BAND = 5.0          # statistical margin (worst observed ~4.2*SE); see T1


def _mc_joint():
    times, w1, w2 = gbm.simulate_correlated_brownian(RHO, 1.0, T, N_STEPS,
                                                     N_PATHS, seed=0)
    emb = H.time_augment_multi(times, w1, w2)
    return H.mc_expected_signature(emb, N)


def test_M_joint_signature_vs_oracle():
    mc, se = _mc_joint()
    oracle = H.correlated_bm_oracle(RHO, T, N)

    # all-level agreement within a 5*SE band on every stochastic coordinate
    stoch = se > 1e-9
    ratio = np.abs(mc - oracle)[stoch] / se[stoch]
    counts = [D ** k for k in range(1, N + 1)]
    offs = np.concatenate([[0], np.cumsum(counts)])[:-1]
    print("\n[M] per-level max |MC - oracle| (3ch, n=1e5):")
    for L, (o, c) in enumerate(zip(offs, counts), start=1):
        print(f"  L{L}: n_coords={c:3d}  max={np.abs(mc - oracle)[o:o + c].max():.2e}")
    print(f"  worst |err|/SE over stochastic coords: {ratio.max():.2f}  "
          f"(band {SE_BAND})")
    assert ratio.max() < SE_BAND


def test_M_cross_coordinate_equals_rho_T():
    mc, se = _mc_joint()
    i12 = signature.word_index((1, 2), D)
    i21 = signature.word_index((2, 1), D)
    cross = mc[i12] + mc[i21]
    band = 4.0 * np.sqrt(se[i12] ** 2 + se[i21] ** 2)
    print(f"\n[M] cross E[S(1,2)+S(2,1)]={cross:.5f}  target rho*T={RHO * T}  "
          f"|err|={abs(cross - RHO * T):.2e}  (4SE band {band:.2e})")
    assert abs(cross - RHO * T) <= band


def test_M_identities_on_three_channel_paths():
    # A2 (shuffle) and A3 (Chen) must hold per-path on the 3-channel joint path.
    times, w1, w2 = gbm.simulate_correlated_brownian(RHO, 1.0, T, 120, 5, seed=1)
    level = 4
    for k in range(w1.shape[0]):
        path = np.stack([times, w1[k], w2[k]], axis=-1)
        s = signature.sig(path, level)

        for u, v in [((1,), (2,)), ((1, 2), (2,)), ((0,), (1, 2))]:
            lhs = H.coeff(s, u, D) * H.coeff(s, v, D)
            rhs = sum(m * H.coeff(s, w, D) for w, m in shuffle.shuffle(u, v).items())
            assert abs(lhs - rhs) < 1e-10

        m = len(path) // 2
        prod = H.tensor_mul_flat(signature.sig(path[:m + 1], level),
                                 signature.sig(path[m:], level), D, level)
        assert np.max(np.abs(prod - s)) < 1e-10
