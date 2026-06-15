"""Forward covector assembly and pricing against an expected signature."""
from __future__ import annotations

import numpy as np

from . import signature


def forward_covector(x0, K, n_channels):
    """Covector for a forward paying ``X_T - K``.

    As a covector over channels ``(time, price)``:
    ``f = (x0 - K)*(empty word) + 1*(price-letter)``. Because ``iisignature``
    omits the empty word, the constant is returned separately from the body
    covector ``vec`` (zeros except ``1`` at the price-letter index ``(1,)``).
    """
    const = x0 - K
    # vec aligns with the flat signature prefix; it is nonzero only at the
    # price-letter (level 1), so we size it to reach that index.
    price_letter = signature.word_index((1,), n_channels)
    vec = np.zeros(price_letter + 1)
    vec[price_letter] = 1.0
    return const, vec


def price(const, vec, expected_sig, r, T):
    """Discounted value ``e^{-rT} * (const*1 + vec . expected_sig)``.

    ``vec`` may be shorter than ``expected_sig`` (the forward only touches the
    level-1 price coordinate); the dot product uses the overlapping prefix.
    """
    expected_sig = np.asarray(expected_sig, dtype=float)
    # TODO(hedging): scale + ridge — signature scaling/normalisation is not
    # needed for a level-1 forward read-off, but will be once the dependency
    # matrix appears in the hedging phase.
    body = np.dot(vec, expected_sig[: vec.shape[0]])
    return np.exp(-r * T) * (const + body)
