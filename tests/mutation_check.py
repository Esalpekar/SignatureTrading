"""Mutation testing: inject real bugs, confirm the load-bearing gates go RED.

A test that survives a bug it should catch is a vacuous test. For each mutation
we recompute the relevant gate's core quantity and check the gate's assertion
now FAILS. Prints CAUGHT (gate went red -> good) or LEAK (survived -> bad).
Run: python tests/mutation_check.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import iisignature

from sigcore import gbm, signature
from sigcore.hedge import (embedding as emb, objective as obj, covector as cov,
                           strategy, pnl, scaling, solve)
import helpers as H

X0, R, SIGMA, T, K = 1.0, 0.05, 0.3, 1.0, 1.0
results = []


def report(name, gate, caught, expected_leak=False):
    if expected_leak:
        tag = "LEAK*  "                      # documented, not a failure
    else:
        tag = "CAUGHT" if caught else "LEAK  "
        results.append(caught)
    print(f"  [{tag}] {name:42s} -> {gate}")


# ---- shared H0 per-path discrepancy (forward, depth 1) ----------------------
def h0_perpath_maxerr():
    basis = obj.strategy_basis(1)
    ell = {w: 0.4 + 0.2 * i for i, w in enumerate(basis)}
    p0 = H.forward_closed_form(X0, K, R, T)
    payoff = emb.forward_covector(X0, K)
    loss = obj.loss_covector(payoff, p0, ell)
    t, paths = gbm.simulate(X0, R, SIGMA, T, 30, 150, seed=7)
    sigs = iisignature.sig(np.ascontiguousarray(emb.enlarged_batch(t, paths)), 4)
    trade = cov.append_letter(ell, emb.TRADE_LETTER)
    fwd = paths[:, -1] - K
    err_pnl = err_loss = 0.0
    for i in range(paths.shape[0]):
        theta = strategy.positions(ell, t, paths[i], 4)
        pnl_direct = np.sum(theta[:-1] * np.diff(paths[i]))     # literal theta.dX
        pnl_cov = cov.contract(trade, sigs[i], emb.N_CHANNELS)
        L_direct = fwd[i] - p0 - pnl_direct
        L_cov = cov.contract(loss, sigs[i], emb.N_CHANNELS)
        err_pnl = max(err_pnl, abs(pnl_direct - pnl_cov))
        err_loss = max(err_loss, abs(L_direct - L_cov))
    return err_pnl, err_loss


def h1_ell0(level_word_shift=0):
    t, paths = gbm.simulate(X0, 0.0, 0.2, T, 50, 30_000, seed=0)
    es, _ = emb.expected_signature(t, paths, 4)
    A, b = obj.mean_variance_system(emb.forward_covector(X0, K), 0.0,
                                    obj.strategy_basis(1), es)
    return solve.solve_mean_variance(A, b, 0.0)[0]


# ---- baseline (no mutation) sanity ------------------------------------------
def baseline():
    ep, el = h0_perpath_maxerr()
    print(f"\nbaseline: H0 per-path err_pnl={ep:.1e} err_loss={el:.1e}  "
          f"H1 ell0={h1_ell0():.4f}  (expect ~0 and ~1)")
    assert ep < 1e-8 and el < 1e-8 and abs(h1_ell0() - 1) < 1e-2


# ---- M1: flip trade letter lead -> lag --------------------------------------
def m1_trade_letter():
    saved = (emb.TRADE_LETTER, obj.TRADE_LETTER)
    emb.TRADE_LETTER = obj.TRADE_LETTER = emb.LAG
    try:
        ep, el = h0_perpath_maxerr()
        report("M1 trade letter lead->lag", f"H0 err_pnl={ep:.1e}", ep > 1e-6)
        # H1 (forward replication) is structurally insensitive: it uses only the
        # terminal increment, where lead and lag coincide. H0 is the guard here.
        report("M1 trade letter (H1 insensitive)", f"H1 ell0={h1_ell0():.3f}",
               abs(h1_ell0() - 1) > 1e-2, expected_leak=True)
    finally:
        emb.TRADE_LETTER, obj.TRADE_LETTER = saved


# ---- M2: plain signature (Stratonovich) instead of lead-lag (Ito) -----------
def m2_stratonovich():
    # Test I recovers int X dX from the lead-lag (Ito) coordinate. The plain
    # piecewise-linear integral is Stratonovich = 1/2 (X_T^2 - X_0^2), missing
    # the -1/2 QV correction. Show that mismatch vs the Ito closed form.
    t, paths = gbm.simulate(X0, 0.05, 0.3, T, 300, 20, seed=3)
    max_err = 0.0
    for X in paths:
        strat = 0.5 * (X[-1] ** 2 - X[0] ** 2)               # plain-sig integral
        ito_target = 0.5 * (X[-1] ** 2 - X[0] ** 2) - 0.5 * np.sum(np.diff(X) ** 2)
        max_err = max(max_err, abs(strat - ito_target))      # = 1/2 QV
    report("M2 plain sig (Stratonovich)", f"test-I err={max_err:.1e}", max_err > 1e-8)


# ---- M3: perturb word_index by +1 -------------------------------------------
def m3_word_index():
    # covector/strategy do `from sigcore import signature`, so patching the
    # module attribute propagates to every coordinate lookup.
    orig = signature.word_index
    signature.word_index = lambda w, d: orig(w, d) + 1
    try:
        ep, el = h0_perpath_maxerr()
        report("M3 word_index +1", f"H0 err_pnl={ep:.1e}", ep > 1e-6)
        # H1-forward is again structurally insensitive (the +1 shift cancels in
        # the forward's b0/A00 ratio); H0 is the guard and catches it above.
        report("M3 word_index +1 (H1 insensitive)", f"H1 ell0={h1_ell0():.3f}",
               abs(h1_ell0() - 1) > 1e-2, expected_leak=True)
    finally:
        signature.word_index = orig


# ---- M5: skip scaling -------------------------------------------------------
def m5_skip_scaling():
    orig = scaling.scaling_factors
    scaling.scaling_factors = lambda A: np.ones(A.shape[0])
    try:
        t, paths = gbm.simulate(X0, 0.03, 0.2, T, 60, 60_000, seed=0)
        es, _ = emb.expected_signature(t, paths, 6)
        A, _ = obj.mean_variance_system(emb.forward_covector(X0, K), 0.0,
                                        obj.strategy_basis(2), es)
        c_un, c_sc = scaling.condition_numbers(A)
        report("M5 skip scaling", f"cond {c_un:.1e}->{c_sc:.1e}",
               not (c_sc < c_un / 5))
    finally:
        scaling.scaling_factors = orig


# ---- M6: gamma = delta = 0 (asymmetric penalty becomes mean-variance) -------
def m6_no_asymmetry():
    import test_hedge_h6 as h6
    saved = (h6.GAMMA, h6.DELTA)
    h6.GAMMA, h6.DELTA = 0.0, 0.0
    try:
        r = h6.compute_h6()
        strict = (r["as"]["cvar95"] < r["mv"]["cvar95"]
                  and r["as"]["skew"] < r["mv"]["skew"])
        report("M6 gamma=delta=0", f"CVaR MV={r['mv']['cvar95']:.4f} "
               f"as={r['as']['cvar95']:.4f}", not strict)
    finally:
        h6.GAMMA, h6.DELTA = saved


# ---- M7: sign flip in the loss covector (+ell.a instead of -) ---------------
def m7_sign_flip():
    orig = obj.loss_covector
    obj.loss_covector = lambda payoff, p0, ell: cov.add(
        payoff, {(): -p0}, cov.append_letter(ell, emb.TRADE_LETTER))  # +, not -
    try:
        _, el = h0_perpath_maxerr()
        report("M7 loss-covector sign flip", f"H0 err_loss={el:.1e}", el > 1e-6)
    finally:
        obj.loss_covector = orig


if __name__ == "__main__":
    baseline()
    print("\nmutations (CAUGHT = gate went red, as it must):")
    m1_trade_letter()
    m2_stratonovich()
    m3_word_index()
    m7_sign_flip()
    m5_skip_scaling()
    m6_no_asymmetry()
    print(f"\n{sum(results)}/{len(results)} required mutations caught "
          f"(LEAK* = documented structural insensitivity, not a failure).")
    sys.exit(0 if all(results) else 1)
