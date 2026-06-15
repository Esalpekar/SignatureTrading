"""H7 — truncation-degree coupling enforced (never silently truncate)."""
import pytest

from sigcore.hedge import embedding as emb, objective as obj


def test_H7_raises_when_signature_too_shallow():
    payoff = emb.forward_covector(1.0, 1.0)
    ell = {w: 0.5 for w in obj.strategy_basis(2)}        # depth 2 -> P&L len 3
    loss = obj.loss_covector(payoff, 0.0, ell)
    need = obj.required_signature_level(loss, 2)          # mean-variance: q=2
    assert need == 6                                      # maxlen(L)=3 x 2

    # available level 4 < needed 6: must raise, not silently drop terms
    with pytest.raises(obj.TruncationError):
        obj.check_truncation(loss, max_power=2, available_level=4)
    # available level 6 is fine
    obj.check_truncation(loss, max_power=2, available_level=6)


def test_H7_quartic_needs_more_depth():
    payoff = emb.forward_covector(1.0, 1.0)
    ell = {w: 0.5 for w in obj.strategy_basis(1)}         # depth 1 -> P&L len 2
    loss = obj.loss_covector(payoff, 0.0, ell)
    # quartic q=4: needs maxlen(L)=2 x 4 = 8
    assert obj.required_signature_level(loss, 4) == 8
    with pytest.raises(obj.TruncationError):
        obj.check_truncation(loss, max_power=4, available_level=6)
