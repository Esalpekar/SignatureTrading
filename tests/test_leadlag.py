"""T2 - Lead-lag area = 1/2 * quadratic variation."""
import numpy as np

from sigcore import gbm, embed, signature


def test_leadlag_area_equals_half_qv():
    times, paths = gbm.simulate(
        x0=1.0, r=0.03, sigma=0.2, T=1.0, n_steps=250, n_paths=20, seed=0
    )

    i01 = signature.word_index((0, 1), 2)
    i10 = signature.word_index((1, 0), 2)

    rel_errs = []
    for series in paths:
        s = signature.sig(embed.lead_lag(series), 2)
        area = 0.5 * (s[i01] - s[i10])              # antisymmetric level-2 part
        qv = np.sum(np.diff(series) ** 2)
        rel_errs.append(abs(area - 0.5 * qv) / (0.5 * qv))

    mean_rel = float(np.mean(rel_errs))
    assert mean_rel < 1e-2, f"mean relative error {mean_rel:.2e} exceeds tolerance"
