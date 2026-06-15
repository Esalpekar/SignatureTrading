"""Shuffle product of words.

The shuffle of two words is the multiset of all interleavings preserving the
internal order of each. It is the exact primitive behind the shuffle identity
(``<S,u>*<S,v> = <S, u shuffle v>``) and the one the hedging phase will reuse.
"""
from __future__ import annotations

from collections import Counter


def shuffle(u, v):
    """Shuffle product of words ``u`` and ``v`` (tuples of channel indices).

    Returns a ``Counter`` mapping each resulting word to its multiplicity::

        shuffle(au, bv) = a * shuffle(u, bv) + b * shuffle(au, v)

    with the empty word as base case.
    """
    u = tuple(u)
    v = tuple(v)
    if not u:
        return Counter({v: 1})
    if not v:
        return Counter({u: 1})

    a, u_rest = u[:1], u[1:]
    b, v_rest = v[:1], v[1:]

    out = Counter()
    for word, mult in shuffle(u_rest, v).items():
        out[a + word] += mult
    for word, mult in shuffle(u, v_rest).items():
        out[b + word] += mult
    return out
