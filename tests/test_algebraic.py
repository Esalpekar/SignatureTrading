"""Category A — algebraic identities.

Exact, per-path, model-free, machine precision. Run on a mix of 2- and
3-channel paths, uniform and irregular grids. If any of these fail the engine
is not computing iterated integrals correctly and nothing downstream is
meaningful. Tolerance for all: < 1e-10.
"""
import itertools

import numpy as np
import pytest

from sigcore import signature, embed, shuffle
import helpers as H

TOL = 1e-10
LEVEL = 4

# (n_points, n_channels, seed, irregular) — the deterministic path zoo.
PATHS = [
    (6, 2, 1, False),
    (7, 2, 2, True),
    (6, 3, 3, False),
    (8, 3, 4, True),
]


def _paths():
    return [H.random_path(n, d, seed, irr) for (n, d, seed, irr) in PATHS]


def test_A1_straight_line_closed_form():
    # Signature of a single segment a->b equals sum_k (b-a)^(x)k / k!.
    for d in (2, 3):
        rng = np.random.default_rng(10 + d)
        a = rng.standard_normal(d)
        b = rng.standard_normal(d)
        path = np.stack([a, b])
        got = signature.sig(path, LEVEL)
        want = H.straight_line_signature(a, b, LEVEL)
        assert np.max(np.abs(got - want)) < TOL


def test_A2_shuffle_identity():
    # <S,u> * <S,v> = <S, u shuffle v> for all words with len(u)+len(v) <= LEVEL.
    for path in _paths():
        d = path.shape[1]
        s = signature.sig(path, LEVEL)
        words = [w for L in range(1, LEVEL) for w in H.words_of_length(L, d)]
        for u, v in itertools.product(words, words):
            if len(u) + len(v) > LEVEL:
                continue
            lhs = H.coeff(s, u, d) * H.coeff(s, v, d)
            rhs = sum(m * H.coeff(s, w, d) for w, m in shuffle.shuffle(u, v).items())
            assert abs(lhs - rhs) < TOL


def test_A3_chen_identity():
    # sig(P) == sig(P[:m+1]) (x) sig(P[m:]) for an interior split m.
    for path in _paths():
        d = path.shape[1]
        m = len(path) // 2
        s = signature.sig(path, LEVEL)
        prod = H.tensor_mul_flat(
            signature.sig(path[: m + 1], LEVEL),
            signature.sig(path[m:], LEVEL),
            d, LEVEL,
        )
        assert np.max(np.abs(prod - s)) < TOL


def test_A4_inverse_time_reversal():
    # sig(P) (x) sig(reverse(P)) == identity (all levels 1..N vanish).
    for path in _paths():
        d = path.shape[1]
        prod = H.tensor_mul_flat(
            signature.sig(path, LEVEL),
            signature.sig(path[::-1], LEVEL),
            d, LEVEL,
        )
        assert np.max(np.abs(prod)) < TOL


def test_A5_reparameterisation():
    # Without a time channel: resampling the same geometric polyline (collinear
    # midpoints inserted) leaves the signature unchanged.
    rng = np.random.default_rng(99)
    spatial = np.cumsum(rng.standard_normal((6, 2)), axis=0) * 0.3
    s = signature.sig(spatial, LEVEL)
    s_resampled = signature.sig(H.subdivide(spatial), LEVEL)
    assert np.max(np.abs(s - s_resampled)) < TOL

    # With time augmented: the SAME spatial vertices on two different increasing
    # time grids give geometrically different (t, x) paths -> signature changes.
    n = len(spatial)
    t1 = np.linspace(0.0, 1.0, n)
    t2 = np.sort(rng.uniform(0.0, 1.0, n))
    t2[0], t2[-1] = 0.0, 1.0
    a = embed.time_augment(t1, spatial[:, 0])
    b = embed.time_augment(t2, spatial[:, 0])
    assert np.max(np.abs(signature.sig(a, LEVEL) - signature.sig(b, LEVEL))) > 1e-6


def test_A6_level1_equals_increments():
    # <S,(i)> = X^i_T - X^i_0 for every channel i.
    for path in _paths():
        d = path.shape[1]
        s = signature.sig(path, LEVEL)
        for i in range(d):
            assert abs(H.coeff(s, (i,), d) - (path[-1, i] - path[0, i])) < TOL


def test_A7_homogeneity():
    for path in _paths():
        d = path.shape[1]
        s = signature.sig(path, LEVEL)
        lam = 1.7

        # scale a single channel i -> word coord scales by lam^(count of i)
        i = d - 1
        scaled = path.copy()
        scaled[:, i] *= lam
        ss = signature.sig(scaled, LEVEL)
        for L in range(1, LEVEL + 1):
            for w in H.words_of_length(L, d):
                factor = lam ** w.count(i)
                assert abs(H.coeff(ss, w, d) - factor * H.coeff(s, w, d)) < TOL

        # scale all channels -> level-k block scales by lam^k
        sall = signature.sig(path * lam, LEVEL)
        off = 0
        for L in range(1, LEVEL + 1):
            cnt = d ** L
            block, ref = sall[off:off + cnt], s[off:off + cnt]
            assert np.max(np.abs(block - (lam ** L) * ref)) < TOL
            off += cnt


def test_A8_degenerate_paths():
    # Constant path: every level-1..N coordinate is zero.
    const = np.ones((5, 2)) * np.array([0.3, -1.2])
    assert np.max(np.abs(signature.sig(const, LEVEL))) < TOL

    # Single-step path matches the A1 straight-line formula.
    a, b = np.array([0.1, 0.2, -0.3]), np.array([1.0, -0.5, 0.7])
    step = np.stack([a, b])
    assert np.max(np.abs(signature.sig(step, LEVEL)
                         - H.straight_line_signature(a, b, LEVEL))) < TOL
