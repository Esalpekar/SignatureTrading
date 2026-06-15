"""Signature computation and the analytic tensor exponential.

Conventions are pinned to ``iisignature``: the flat signature vector holds
levels ``1..N`` (the empty word / level 0 is omitted), ordered level by level,
and lexicographically over channel indices within each level. ``tensor_exp``
and ``word_index`` must produce output coordinate-aligned with that.
"""
from __future__ import annotations

import numpy as np
import iisignature


def sig(path, level):
    """Flat level ``1..level`` signature of ``path`` via ``iisignature.sig``.

    ``path`` has shape ``(n_points, n_channels)``.
    """
    return iisignature.sig(np.asarray(path, dtype=float), level)


def word_index(word, n_channels):
    """Position of ``word`` (a tuple of channel indices) in the flat signature.

    Explicit rather than hard-coded so callers can index by meaning, e.g.
    the price-letter ``(1,)``. Levels below the word's length are skipped, then
    the lexicographic offset within the word's own level is added.
    """
    word = tuple(word)
    L = len(word)
    if L == 0:
        raise ValueError("empty word is not stored (iisignature omits level 0)")

    offset = sum(n_channels ** k for k in range(1, L))      # full lower levels
    within = 0
    for c in word:
        within = within * n_channels + c                    # lexicographic
    return offset + within


# --- truncated tensor algebra -------------------------------------------------
#
# An element is represented as a list ``comp`` where ``comp[k]`` is the level-k
# tensor of shape ``(d,) * k`` (``comp[0]`` is a scalar 0-d array). Truncated
# multiplication concatenates words and drops anything above ``level``.


def _zero(d, level):
    return [np.zeros((d,) * k) for k in range(level + 1)]


def _identity(d, level):
    e = _zero(d, level)
    e[0] = np.array(1.0)
    return e


def _mul(a, b, d, level):
    """Truncated tensor product of two algebra elements."""
    out = _zero(d, level)
    for i in range(level + 1):
        if not a[i].any():
            continue
        for j in range(level + 1 - i):
            if not b[j].any():
                continue
            # outer product -> level (i+j) tensor, in lexicographic word order
            prod = np.tensordot(a[i], b[j], axes=0)
            out[i + j] = out[i + j] + prod
    return out


def _flatten(comp, level):
    """Flatten levels ``1..level`` in iisignature order (level 0 omitted)."""
    return np.concatenate([comp[k].ravel() for k in range(1, level + 1)])


def tensor_exp(generator, n_channels, level):
    """Truncated tensor exponential ``sum_{k=0}^{level} A^{(x)k} / k!``.

    ``generator`` is ``(g1, g2)``: the level-1 component ``g1`` (shape
    ``(d,)``) and the level-2 component ``g2`` (shape ``(d, d)``). Output is
    flattened in iisignature's word order (level 0 omitted) so it is
    coordinate-aligned with :func:`sig`.
    """
    d = n_channels
    g1, g2 = generator
    A = _zero(d, level)
    A[1] = np.asarray(g1, dtype=float)
    if level >= 2:
        A[2] = np.asarray(g2, dtype=float)

    result = _identity(d, level)
    power = _identity(d, level)
    for k in range(1, level + 1):
        power = _mul(power, A, d, level)
        kfac = _factorial(k)
        for m in range(level + 1):
            result[m] = result[m] + power[m] / kfac

    return _flatten(result, level)


def _factorial(k):
    f = 1
    for i in range(2, k + 1):
        f *= i
    return f
