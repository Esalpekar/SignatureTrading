"""Category E — embedding & numerical edge cases."""
import numpy as np

from sigcore import gbm, heston, embed, signature
import helpers as H

TOL = 1e-10
LEVEL = 4


def test_E1_channel_order():
    # Time is channel 0 everywhere: the time coord = T and the price coord
    # = X_T - X_0 (this is exactly the A6 mapping, used here to pin order).
    times, paths = gbm.simulate(1.0, 0.03, 0.2, 1.0, 50, 1, seed=0)
    path = embed.time_augment(times, paths[0])
    s = signature.sig(path, LEVEL)
    assert abs(H.coeff(s, (0,), 2) - times[-1]) < TOL
    assert abs(H.coeff(s, (1,), 2) - (paths[0, -1] - paths[0, 0])) < TOL


def test_E2_leadlag_shape_and_alignment():
    x = np.array([1.0, 1.5, 0.7, 2.0, 1.2])
    ll = embed.lead_lag(x)
    assert ll.shape == (2 * len(x) - 1, 2)
    # at each odd node the lead has already advanced while the lag still trails
    for k in range(len(x) - 1):
        assert ll[2 * k + 1, 0] == x[k + 1]   # lead leads
        assert ll[2 * k + 1, 1] == x[k]       # lag trails by one sample
    # endpoints
    assert ll[0, 0] == x[0] and ll[0, 1] == x[0]
    assert ll[-1, 0] == x[-1] and ll[-1, 1] == x[-1]


def test_E3_irregular_grid_identities():
    # All Category-A identities must survive a non-uniform time grid.
    path = H.random_path(8, 2, seed=7, irregular=True)
    d = 2
    s = signature.sig(path, LEVEL)

    # A6 increments
    for i in range(d):
        assert abs(H.coeff(s, (i,), d) - (path[-1, i] - path[0, i])) < TOL
    # A3 Chen
    m = len(path) // 2
    prod = H.tensor_mul_flat(signature.sig(path[:m + 1], LEVEL),
                             signature.sig(path[m:], LEVEL), d, LEVEL)
    assert np.max(np.abs(prod - s)) < TOL
    # A4 inverse
    inv = H.tensor_mul_flat(s, signature.sig(path[::-1], LEVEL), d, LEVEL)
    assert np.max(np.abs(inv)) < TOL


def test_E4_determinism():
    a = gbm.simulate(1.0, 0.03, 0.2, 1.0, 100, 1000, seed=42)
    b = gbm.simulate(1.0, 0.03, 0.2, 1.0, 100, 1000, seed=42)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    h1 = heston.simulate(1.0, 0.04, 0.03, 2.0, 0.04, 0.3, -0.7, 1.0, 100, 500, seed=7)
    h2 = heston.simulate(1.0, 0.04, 0.03, 2.0, 0.04, 0.3, -0.7, 1.0, 100, 500, seed=7)
    for u, v in zip(h1, h2):
        assert np.array_equal(u, v)


def test_E5_magnitude_sanity():
    # No inf/nan in signatures of either model; magnitudes stay in a sane range
    # for this scale (early warning for the deferred scaling work).
    times, paths = gbm.simulate(1.0, 0.03, 0.2, 1.0, 250, 200, seed=0)
    sg = H.mc_expected_signature(embed.time_augment(times, paths), LEVEL)[0]
    assert np.all(np.isfinite(sg))
    assert np.max(np.abs(sg)) < 1e3

    th, hp, _ = heston.simulate(1.0, 0.04, 0.03, 2.0, 0.04, 0.3, -0.7,
                                1.0, 250, 200, seed=0)
    sh = H.mc_expected_signature(H.log_price_embed(th, hp), LEVEL)[0]
    assert np.all(np.isfinite(sh))
    assert np.max(np.abs(sh)) < 1e3
