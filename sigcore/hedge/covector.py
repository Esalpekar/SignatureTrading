"""Covector algebra over words (dicts ``{word: coeff}``, ``()`` = empty word).

A covector pairs with a signature by ``<c, S> = sum_w c_w * S_w`` with the
empty word contributing ``c_() * 1``. These are the operations the objective is
built from: shuffle product (turns products of linear functionals into a single
covector), letter append (the trade integral), and contraction with ``E[S]``.
"""
from __future__ import annotations

from sigcore import signature, shuffle


def add(*covs):
    out = {}
    for c in covs:
        for w, v in c.items():
            out[w] = out.get(w, 0.0) + v
    return out


def scale(cov, alpha):
    return {w: alpha * v for w, v in cov.items()}


def append_letter(cov, letter):
    """Append ``letter`` to every word — the trade-letter operation ``c -> c.a``."""
    return {w + (letter,): v for w, v in cov.items()}


def shuffle_cov(c1, c2):
    """Shuffle product of two covectors (bilinear extension of word shuffle)."""
    out = {}
    for w1, v1 in c1.items():
        for w2, v2 in c2.items():
            for w, m in shuffle.shuffle(w1, w2).items():
                out[w] = out.get(w, 0.0) + v1 * v2 * m
    return out


def shuffle_power(cov, k):
    """``cov^{shuffle k}`` (the empty word covector for ``k = 0``)."""
    res = {(): 1.0}
    for _ in range(k):
        res = shuffle_cov(res, cov)
    return res


def max_word_length(cov):
    return max((len(w) for w in cov), default=0)


def contract(cov, expected_sig, n_channels):
    """``<cov, E[S]>``; empty word pairs with 1, others via word_index."""
    total = 0.0
    for w, v in cov.items():
        if len(w) == 0:
            total += v
        else:
            total += v * expected_sig[signature.word_index(w, n_channels)]
    return total
