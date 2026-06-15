"""Foresight — conditioning probe for the hedging dependency matrix.

Assembles the Gram/dependency matrix the hedging least-squares will invert:
``A_{wv} = <(w + trade) shuffle (v + trade), E[S]>``, where ``trade`` is the
price-letter (a ``dX``). No hard pass/fail on the magnitude (per the v2 spec) —
this is the early-warning instrument for the estimation wall: ``cond(A)`` blows
up with truncation level. The test only asserts the machinery assembles and the
numbers are finite and *grow* with level, so the probe itself can't silently
break. The actual condition numbers are printed by ``report.py``.
"""
import numpy as np

from sigcore import signature, shuffle
import helpers as H

R, SIGMA, T, D = 0.03, 0.2, 1.0, 2
TRADE = (1,)


def _coeff(es, word):
    return 1.0 if len(word) == 0 else es[signature.word_index(word, D)]


def _pair(es, wa, wb):
    return sum(m * _coeff(es, w) for w, m in shuffle.shuffle(wa, wb).items())


def dependency_matrix(level):
    """Gram matrix over trading words up to ``level`` (plus the constant)."""
    es = H.gbm_log_expected_signature(R, SIGMA, T, 2 * level + 2)
    basis = [()] + [w for k in range(1, level + 1)
                    for w in H.words_of_length(k, D)]
    n = len(basis)
    A = np.empty((n, n))
    for i, wi in enumerate(basis):
        for j, wj in enumerate(basis):
            A[i, j] = _pair(es, wi + TRADE, wj + TRADE)
    return A


def test_conditioning_probe_assembles_and_grows():
    conds = []
    for level in (1, 2, 3):
        A = dependency_matrix(level)
        assert np.all(np.isfinite(A))
        c = np.linalg.cond(A)
        assert np.isfinite(c)
        conds.append(c)
    # the whole point: conditioning degrades with level (watch it now, before
    # it is a surprise in the hedging phase).
    assert conds[0] < conds[1] < conds[2]
