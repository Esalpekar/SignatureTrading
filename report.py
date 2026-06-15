"""Validation report: runs all four checks, prints a motivated plain-text
report, and exits non-zero if any check fails (so it can gate CI later).

Run:  python report.py
"""
from __future__ import annotations

import sys
import os
import itertools
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from math import comb, factorial

import numpy as np
import iisignature

from sigcore import gbm, embed, signature, price, heston, shuffle
from sigcore import expected_signature as es

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- shared parameters ------------------------------------------------------
X0 = 1.0
R = 0.03
SIGMA = 0.2
T = 1.0
N_STEPS = 250
N_PATHS = 100_000
LEVEL = 3
SEED = 0
D = 2

# T1/T3 use the *unit* Brownian driver (sigma=1), the one process with a fully
# known expected signature (Fawcett). SIGMA above is the GBM vol, used by T2/T4.
SIGMA_BM = 1.0


def _brownian_analytic():
    """Analytic expected signature of the (t, W_t) Brownian driver, level 3."""
    g1 = np.array([T, 0.0])
    g2 = np.array([[0.0, 0.0], [0.0, T / 2.0]])
    return es.analytic_gaussian((g1, g2), D, LEVEL)


def t1():
    print("[T1] Signature engine vs Fawcett  (Brownian motion, seed sweep)")
    print("  why: the only process with a fully known expected signature; "
          "checks the engine itself.")
    analytic = _brownian_analytic()
    i = lambda w: signature.word_index(w, D)

    worst_ratio, worst_abs, ok = 0.0, 0.0, True
    seed0 = None
    for seed in range(8):
        times, W = gbm.simulate_brownian(SIGMA_BM, T, N_STEPS, N_PATHS, seed=seed)
        mc, se = _mc_sig(embed.time_augment(times, W), LEVEL)
        if seed == 0:
            seed0 = mc
        err = np.abs(mc - analytic)
        stoch = se > 1e-9
        ratio = float((err[stoch] / se[stoch]).max())
        worst_ratio = max(worst_ratio, ratio)
        worst_abs = max(worst_abs, float(err.max()))
        ok = ok and ratio < 5.0

    # representative coordinate values from seed 0
    print(f"  E[int dt]    {seed0[i((0,))]:.4f}  target {analytic[i((0,))]:.4f}")
    print(f"  E[int dW]    {seed0[i((1,))]:.4f}  target {analytic[i((1,))]:.4f}")
    print(f"  E[(1,1)]     {seed0[i((1,1))]:.4f}  target {analytic[i((1,1))]:.4f}")
    # The fixed 5e-3 single-seed tol was ~98% used at seed 0 and is exceeded by
    # other seeds (worst max|err|~7.4e-3): a flaky gate. Gate on a 5*SE band.
    print(f"  worst |err|/SE over 8 seeds: {worst_ratio:.2f}  (band 5.0)   "
          f"[worst max|err|={worst_abs:.1e}, fixed-5e-3 would be flaky]   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def t2():
    print("[T2] Lead-lag area = 1/2 * quadratic variation")
    print("  why: the embedding that will carry Ito trading P&L in the "
          "hedging phase.")

    _, paths = gbm.simulate(X0, R, SIGMA, T, N_STEPS, 20, seed=SEED)
    i01 = signature.word_index((0, 1), D)
    i10 = signature.word_index((1, 0), D)

    rel = []
    for series in paths:
        s = signature.sig(embed.lead_lag(series), 2)
        area = 0.5 * (s[i01] - s[i10])
        qv = np.sum(np.diff(series) ** 2)
        rel.append(abs(area - 0.5 * qv) / (0.5 * qv))
    mean_rel = float(np.mean(rel))

    tol = 1e-2
    ok = mean_rel < tol
    print(f"  mean relative error: {mean_rel:.1e}                              "
          f"   {'PASS' if ok else 'FAIL'}")
    return ok


def t3():
    print("[T3] Monte-Carlo convergence of E[S]")
    print("  why: confirms estimation error decays like 1/sqrt(n), not with "
          "grid size.")

    analytic = _brownian_analytic()
    ns = [100, 1_000, 10_000, 100_000]
    errs = []
    for n in ns:
        times, W = gbm.simulate_brownian(SIGMA_BM, T, N_STEPS, n, seed=SEED)
        mc, _ = es.monte_carlo(embed.time_augment(times, W), LEVEL)
        errs.append(float(np.linalg.norm(mc - analytic)))

    print("  " + "   ".join(
        f"n={n:.0e} err={e:.2e}" for n, e in zip(ns, errs)
    ))
    slope = float(np.polyfit(np.log(ns), np.log(errs), 1)[0])
    ok = -0.6 <= slope <= -0.4
    print(f"  log-log slope: {slope:.2f}                                       "
          f"   {'PASS' if ok else 'FAIL'}")
    return ok


def t4():
    print("[T4] Forward price vs closed form  X0 - K*e^{-rT}")
    print("  why: end-to-end -- covector, indexing, discounting, expected "
          "signature combine correctly.")

    K = 1.0
    closed_form = X0 - K * np.exp(-R * T)
    const, vec = price.forward_covector(X0, K, D)
    pidx = signature.word_index((1,), D)

    es_an = np.zeros(pidx + 1)
    es_an[pidx] = X0 * (np.exp(R * T) - 1.0)
    p_an = price.price(const, vec, es_an, R, T)
    err_an = abs(p_an - closed_form)
    ok_an = err_an < 1e-10

    times, paths = gbm.simulate(X0, R, SIGMA, T, N_STEPS, N_PATHS, seed=SEED)
    mc, se = es.monte_carlo(embed.time_augment(times, paths), LEVEL)
    p_mc = price.price(const, vec, mc, R, T)
    err_mc = abs(p_mc - closed_form)
    band = 2.0 * np.exp(-R * T) * se[pidx]
    ok_mc = err_mc <= band

    print(f"  closed form     : {closed_form:.6f}")
    print(f"  analytic pipe   : {p_an:.6f}   |err| {err_an:.1e}                "
          f"          {'PASS' if ok_an else 'FAIL'}")
    print(f"  monte-carlo pipe: {p_mc:.6f}   |err| {err_mc:.1e}   "
          f"(MC band {band:.1e})      {'PASS' if ok_mc else 'FAIL'}")
    return ok_an and ok_mc


# ---- v1 comprehensive blocks (B: depth, D: Heston) --------------------------
#
# These extend the v0 forward-only report to characterise the estimation error
# at every level and to bring in Heston, mirroring the v1 test spec. They print
# numbers (per-level tables, sampling-vs-bias breakdown, Heston variance
# stability), not just pass/fail.

B1_THRESH = [3.0e-3, 2.0e-3, 1.0e-3, 5.0e-4]      # calibrated, levels 1..4
LEVEL_B = 4
_COUNTS = [2 ** L for L in range(1, LEVEL_B + 1)]
_OFFSETS = np.concatenate([[0], np.cumsum(_COUNTS)])[:-1]

# Heston base params (Feller-satisfying: 2*kappa*theta = 0.16 >= xi^2 = 0.09)
H_V0, H_KAPPA, H_THETA, H_XI, H_RHO = 0.04, 2.0, 0.04, 0.3, -0.7


def _gbm_log_oracle(sigma, level):
    g1 = np.array([T, (R - 0.5 * sigma * sigma) * T])
    g2 = np.zeros((2, 2))
    g2[1, 1] = 0.5 * sigma * sigma * T
    return signature.tensor_exp((g1, g2), D, level)


def _mc_sig(embedded, level):
    """Batched mean signature and per-coordinate standard error."""
    sigs = iisignature.sig(np.ascontiguousarray(embedded, dtype=float), level)
    return sigs.mean(axis=0), sigs.std(axis=0, ddof=1) / np.sqrt(sigs.shape[0])


def b_block():
    print("[B]  Expected-signature accuracy & error budget  (log-price GBM)")
    print("  why: certifies the signature above level 1 and splits the error "
          "into sampling vs grid.")

    # B1/B2 -- per-level error and standard-error table at the headline params.
    times, paths = gbm.simulate(X0, R, SIGMA, T, N_STEPS, N_PATHS, seed=SEED)
    mc, se = _mc_sig(embed.time_augment(times, np.log(paths)), LEVEL_B)
    oracle = _gbm_log_oracle(SIGMA, LEVEL_B)
    err = np.abs(mc - oracle)

    print("  level | n_coords |  max|err| | mean|err| |  mean SE | within-4SE | thr")
    ok = True
    for L, (o, c) in enumerate(zip(_OFFSETS, _COUNTS), start=1):
        eblk, sblk = err[o:o + c], se[o:o + c]
        stoch = sblk > 1e-9
        w4 = np.mean(eblk[stoch] <= 4 * sblk[stoch]) if stoch.any() else 1.0
        thr = B1_THRESH[L - 1]
        passed = eblk.max() < thr
        ok = ok and passed
        print(f"   L{L}   |   {c:2d}     |  {eblk.max():.2e} |  {eblk.mean():.2e} "
              f"| {sblk.mean():.2e} |    {w4 * 100:3.0f}%    | {thr:.1e}"
              f"  {'PASS' if passed else 'FAIL'}")

    # B3 -- sampling-error scaling: SE(4n)/SE(n) ~ 0.5.
    _, se_n = _mc_sig(embed.time_augment(*_logsim(25_000)), LEVEL_B)
    _, se_4n = _mc_sig(embed.time_augment(*_logsim(100_000)), LEVEL_B)
    ratios = [se_4n[o + c - 1] / se_n[o + c - 1] for o, c in zip(_OFFSETS, _COUNTS)]
    b3_ok = all(0.425 <= r <= 0.575 for r in ratios)
    print("  B3 sampling-error scaling  SE(4n)/SE(n) per level (target 0.50): "
          + "  ".join(f"L{i+1}={r:.2f}" for i, r in enumerate(ratios))
          + f"   {'PASS' if b3_ok else 'FAIL'}")

    # B4 -- discretisation bias, isolated by common-random-number subsampling.
    bias = _b4_bias()
    b4_ok = all(b <= a * 1.05 for a, b in zip(bias, bias[1:])) and bias[0] > 2 * bias[-1]
    print("  B4 discretisation bias  ||err lvl>=2|| vs n_steps (CRN, sigma=0.4): "
          + "  ".join(f"{ns}:{b:.1e}" for ns, b in zip([10, 50, 125, 250, 500], bias))
          + f"   {'PASS' if b4_ok else 'FAIL'}")
    print("  read: B3 falls like 1/sqrt(n) (sampling); B4 falls with the grid "
          "(bias). Separate floors.")

    return ok and b3_ok and b4_ok


def _logsim(n_paths):
    times, paths = gbm.simulate(X0, R, SIGMA, T, N_STEPS, n_paths, seed=SEED)
    return times, np.log(paths)


def _b4_bias():
    sigma, n_paths, fine, level = 0.4, 100_000, 500, 3
    oracle = _gbm_log_oracle(sigma, level)
    rng = np.random.default_rng(SEED)
    dt_f = T / fine
    incr = ((R - 0.5 * sigma ** 2) * dt_f
            + sigma * np.sqrt(dt_f) * rng.standard_normal((n_paths, fine)))
    out = []
    for ns in [10, 50, 125, 250, 500]:
        agg = incr.reshape(n_paths, ns, fine // ns).sum(axis=2)
        logp = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(agg, axis=1)], axis=1)
        emb = embed.time_augment(np.linspace(0.0, T, ns + 1), logp)
        mc, _ = _mc_sig(emb, level)
        out.append(float(np.linalg.norm((mc - oracle)[2:])))
    return out


def d_block():
    print("[D]  Heston — integrated variance & higher-order stability")
    print("  why: confirms the engine survives stochastic vol and that "
          "higher-order coords are low-variance, absent a full oracle.")

    times, hp, var = heston.simulate(X0, H_V0, R, H_KAPPA, H_THETA, H_XI, H_RHO,
                                     T, N_STEPS, N_PATHS, seed=SEED)

    # D3 -- integrated variance via lead-lag of log-price (closed form).
    area = 0.5 * np.sum(np.diff(np.log(hp), axis=1) ** 2, axis=1)
    mean_a, se_a = area.mean(), area.std(ddof=1) / np.sqrt(len(area))
    tgt = 0.5 * (H_THETA * T + (H_V0 - H_THETA) * (1 - np.exp(-H_KAPPA * T)) / H_KAPPA)
    d3_ok = abs(mean_a - tgt) <= 4 * se_a
    print(f"  D3 integrated variance  area={mean_a:.6f}  target={tgt:.6f}  "
          f"|err|={abs(mean_a - tgt):.1e}  (4SE band {4 * se_a:.1e})   "
          f"{'PASS' if d3_ok else 'FAIL'}")

    # D4 -- relative SE by level; name the worst level-3 coord (above floor).
    mc, se = _mc_sig(embed.time_augment(times, np.log(hp)), 3)
    print("  D4 relative SE by level (mean over coords):")
    for L, (o, c) in enumerate(zip(_OFFSETS[:3], _COUNTS[:3]), start=1):
        m, s = mc[o:o + c], se[o:o + c]
        rel = np.where(np.abs(m) > 1e-14, s / np.abs(m), 0.0)
        print(f"     L{L}: mean relSE = {100 * rel.mean():.1f}%")
    blk = slice(6, 14)
    m, s = mc[blk], se[blk]
    floor = 0.01 * np.linalg.norm(m)
    rel = np.where(np.abs(m) > 1e-14, s / np.abs(m), 0.0)
    above = np.abs(m) > floor
    worst = np.argmax(np.where(above, rel, -1))
    words3 = list(itertools.product(range(2), repeat=3))
    d4_ok = rel[above].max() < 0.10
    print(f"     worst level-3 relSE (signal coords): {100 * rel[above].max():.1f}% "
          f"at word {words3[worst]}  (threshold 10%)   {'PASS' if d4_ok else 'FAIL'}")
    print("  caveat: this certifies the engine survives stochastic vol and its "
          "moments are right — NOT that")
    print("          signature pricing reproduces Heston option prices (where "
          "Heston differs from GBM); that needs the regression path, unbuilt.")

    return d3_ok and d4_ok


# ---- U: truncation / universality convergence -------------------------------

def _central_moment(k):
    return sum(comb(k, j) * (-X0) ** (k - j)
               * X0 ** j * np.exp(j * (R - 0.5 * SIGMA ** 2) * T
                                  + 0.5 * j * j * SIGMA ** 2 * T)
               for j in range(k + 1))


def _taylor_coeff(p, k):
    num = 1.0
    for i in range(k):
        num *= (p - i)
    return num / factorial(k) * X0 ** (p - k)


def _shuffle_power(letter, k):
    res = {(): 1}
    for _ in range(k):
        nxt = {}
        for w, m in res.items():
            for w2, m2 in shuffle.shuffle(w, letter).items():
                nxt[w2] = nxt.get(w2, 0) + m * m2
        res = nxt
    return res


def u_block():
    print("[U]  Universality / truncation convergence  (power claim X_T^p)")
    print("  why: forward & Asian are exact signature coords; only a "
          "non-representable payoff tests the approximation.")
    nmax, d, letter, tol = 5, 2, (1,), 1e-4
    siglen = sum(d ** k for k in range(1, nmax + 1))
    es_vec = np.zeros(siglen)
    for k in range(1, nmax + 1):
        es_vec[signature.word_index(letter * k, d)] = _central_moment(k) / factorial(k)

    ok = True
    for p in (1.5, 0.5):
        closed = np.exp(-R * T) * X0 ** p * np.exp(p * (R - 0.5 * SIGMA ** 2) * T
                                                   + 0.5 * p * p * SIGMA ** 2 * T)
        errs = []
        for n in range(1, nmax + 1):
            const = _taylor_coeff(p, 0)
            vec = np.zeros(siglen)
            for k in range(1, n + 1):
                ck = _taylor_coeff(p, k)
                for w, m in _shuffle_power(letter, k).items():
                    vec[signature.word_index(w, d)] += ck * m
            errs.append(abs(price.price(const, vec, es_vec, R, T) - closed))
        monotone = all(b < a for a, b in zip(errs, errs[1:]))
        passed = monotone and errs[-1] < tol
        ok = ok and passed
        print(f"  X_T^{p}: " + " ".join(f"N{n}={e:.1e}" for n, e in enumerate(errs, 1))
              + f"   {'PASS' if passed else 'FAIL'}")

    # path-dependent stretch: exp(running average) — populates the MIXED (1,0)
    # shuffle-power coords (X_T^p only touches the endpoint slice). No closed
    # form, so converge to the direct MC price on the same paths.
    s_ok = _u_path_dependent()
    ok = ok and s_ok
    return ok


def _u_path_dependent():
    nmax, mixed = 4, (1, 0)
    times, paths = gbm.simulate(X0, R, SIGMA, T, 100, 20_000, seed=1)
    sigs = iisignature.sig(
        np.ascontiguousarray(embed.time_augment(times, paths), dtype=float), 2 * nmax)
    es_mc = sigs.mean(axis=0)
    i10 = signature.word_index(mixed, D)
    avg = X0 + (1.0 / T) * sigs[:, i10]
    direct = np.exp(-R * T) * np.mean(np.exp(avg))

    errs = []
    for n in range(1, nmax + 1):
        const = np.exp(X0)
        vec = np.zeros(es_mc.shape[0])
        for k in range(1, n + 1):
            ck = np.exp(X0) / factorial(k) * (1.0 / T) ** k
            for w, m in _shuffle_power(mixed, k).items():
                vec[signature.word_index(w, D)] += ck * m
        errs.append(abs(price.price(const, vec, es_mc, R, T) - direct))
    ok = all(b < a for a, b in zip(errs, errs[1:])) and errs[-1] < 1e-4
    print(f"  exp(avg) [path-dependent, mixed coords] vs direct MC={direct:.4f}: "
          + " ".join(f"N{n}={e:.1e}" for n, e in enumerate(errs, 1))
          + f"   {'PASS' if ok else 'FAIL'}")
    return ok


def m_block():
    print("[M]  Multi-asset joint signature  (correlated (t, W1, W2))")
    print("  why: cross-asset mixed-letter coords (sign/ordering) are unrun by "
          "any single-asset test; the engine for them exists today.")
    rho, level = 0.5, 4
    times, w1, w2 = gbm.simulate_correlated_brownian(rho, 1.0, T, N_STEPS,
                                                     N_PATHS, seed=SEED)
    emb = np.stack([np.broadcast_to(times, w1.shape), w1, w2], axis=-1)
    mc, se = _mc_sig(emb, level)
    oracle = _correlated_oracle(rho, level)
    stoch = se > 1e-9
    worst = float((np.abs(mc - oracle)[stoch] / se[stoch]).max())
    i12 = signature.word_index((1, 2), 3)
    i21 = signature.word_index((2, 1), 3)
    cross = mc[i12] + mc[i21]
    band = 4.0 * np.sqrt(se[i12] ** 2 + se[i21] ** 2)
    ok = worst < 5.0 and abs(cross - rho * T) <= band
    print(f"  all-level worst |err|/SE: {worst:.2f} (band 5.0)")
    print(f"  cross E[S(1,2)+S(2,1)]={cross:.5f}  target rho*T={rho * T}  "
          f"|err|={abs(cross - rho * T):.1e}  (4SE {band:.1e})   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _correlated_oracle(rho, level):
    g1 = np.array([T, 0.0, 0.0])
    g2 = np.zeros((3, 3))
    g2[1, 1] = g2[2, 2] = 0.5 * T
    g2[1, 2] = g2[2, 1] = 0.5 * rho * T
    return signature.tensor_exp((g1, g2), 3, level)


# ---- I: state-dependent Ito recovery ----------------------------------------

def i_block():
    print("[I]  Ito recovery, state-dependent integrand  (int X dX via lead-lag)")
    print("  why: T2/D3 only cover a constant integrand (QV); the hedge runs "
          "on a varying one.")
    times, paths = gbm.simulate(X0, 0.05, 0.3, T, 300, 50, seed=3)
    i_ll = signature.word_index((1, 0), 2)
    max_err = 0.0
    for series in paths:
        s = signature.sig(embed.lead_lag(series), 2)
        recovered = s[i_ll] + series[0] * (series[-1] - series[0])
        exact = 0.5 * (series[-1] ** 2 - series[0] ** 2) - 0.5 * np.sum(np.diff(series) ** 2)
        max_err = max(max_err, abs(recovered - exact))
    ok = max_err < 1e-8
    print(f"  max per-path |recovered - exact int X dX|: {max_err:.1e}   "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ---- conditioning probe (report-only early warning) -------------------------

def cond_block():
    print("[*]  Conditioning probe (no pass/fail — early warning for hedging)")
    print("  why: the hedging least-squares inverts this Gram matrix; watch "
          "cond(A) blow up with level now.")
    print("  note: thermometer, not cure — it shows the wall EXISTS; that ridge "
          "gets through it is the hedging phase's to prove.")
    trade = (1,)

    # Built from the Monte-Carlo E[S] (deep enough for the shuffles), i.e. the
    # noisy estimate hedging will actually face — not the clean oracle.
    times, paths = gbm.simulate(X0, R, SIGMA, T, N_STEPS, 50_000, seed=SEED)
    es_mc, _ = _mc_sig(embed.time_augment(times, np.log(paths)), 8)

    def coeff(es_vec, word):
        return 1.0 if not word else es_vec[signature.word_index(word, 2)]

    for level in (1, 2, 3):
        basis = [()] + [w for k in range(1, level + 1)
                        for w in itertools.product(range(2), repeat=k)]
        n = len(basis)
        A = np.empty((n, n))
        for i, wi in enumerate(basis):
            for j, wj in enumerate(basis):
                A[i, j] = sum(m * coeff(es_mc, w)
                              for w, m in shuffle.shuffle(wi + trade, wj + trade).items())
        print(f"  level {level} (MC E[S]): basis={n:2d}  cond(A)={np.linalg.cond(A):.2e}")
    return True


# ---- category inventory: run the full pytest suite, enumerate every group ---

_CATEGORIES = [
    ("test_engine.py",        "[T1] engine vs Fawcett"),
    ("test_leadlag.py",       "[T2] lead-lag area = 1/2 QV"),
    ("test_convergence.py",   "[T3] MC convergence"),
    ("test_forward.py",       "[T4/C1] forward price"),
    ("test_algebraic.py",     "[A]  algebraic identities A1-A8"),
    ("test_expected_sig.py",  "[B]  expected-sig depth B1-B5"),
    ("test_pricing.py",       "[C]  pricing C2-C4"),
    ("test_heston.py",        "[D]  Heston D1-D5"),
    ("test_embedding_edge.py", "[E]  embedding/edge E1-E5"),
    ("test_universality.py",  "[U]  truncation convergence"),
    ("test_ito.py",           "[I]  state-dependent Ito"),
    ("test_multiasset.py",    "[M]  multi-asset joint signature"),
    ("test_conditioning.py",  "[*]  conditioning probe"),
    ("test_hedge_objective.py", "[H0] hedge objective bedrock"),
    ("test_hedge_solve.py",   "[H1/H5] replication + convex"),
    ("test_hedge_scaling.py", "[H4] scaling / ridge"),
    ("test_hedge_h2.py",      "[H2] level-0 = cov ratio"),
    ("test_hedge_pnl.py",     "[H3/H8] var reduction + OOS"),
    ("test_hedge_h6.py",      "[H6] asymmetric tail (headline)"),
    ("test_hedge_h7.py",      "[H7] truncation coupling"),
    ("test_multiasset_engine.py", "[M0/M-I] joint engine + cross-Ito"),
    ("test_multiasset_hedge.py", "[MH0-3,5] multi-asset hedging"),
    ("test_multiasset_h4.py",  "[MH4] cond + pairs/ridge"),
    ("test_multiasset_h6.py",  "[MH6] multi-asset asym tail"),
]


def category_inventory():
    print("[INVENTORY] full pytest suite — every category must exist and pass")
    print("  why: a report that prints a subset cannot certify the whole; this "
          "enumerates every group so none can hide.")
    xml_path = tempfile.mktemp(suffix=".xml")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider",
         "--junit-xml", xml_path, "tests"],
        cwd=HERE, capture_output=True, text=True,
    )
    stems = [fname[:-3] for fname, _ in _CATEGORIES]   # e.g. "test_algebraic"
    counts = {}
    try:
        xml_ok = True
        root = ET.parse(xml_path).getroot()
        for tc in root.iter("testcase"):
            # Attribute each testcase to a category by matching a known file stem
            # anywhere in its file/classname/name. Collection errors (where the
            # classname is the module path and there is an <error> child) are
            # attributed the same way, so a broken file FAILs *its* category line.
            src = " ".join(filter(None, [tc.get("file", ""), tc.get("classname", ""),
                                         tc.get("name", "")]))
            fname = next((s + ".py" for s in stems if s in src), "?")
            ok = not any(c.tag in ("failure", "error") for c in tc)
            p, f = counts.get(fname, (0, 0))
            counts[fname] = (p + (1 if ok else 0), f + (0 if ok else 1))
    except ET.ParseError:
        xml_ok = False                                  # malformed junit -> fail hard
    finally:
        if os.path.exists(xml_path):
            os.remove(xml_path)

    all_ok = (proc.returncode == 0) and xml_ok
    listed = set()
    for fname, label in _CATEGORIES:
        p, f = counts.get(fname, (0, 0))
        listed.add(fname)
        total = p + f
        # zero-collected is a FAIL: a renamed/emptied/removed file would
        # otherwise print PASS 0/0 — a false all-clear exactly where it matters.
        if total == 0:
            status, note = "FAIL", "  <-- collected 0 tests"
        elif f == 0:
            status, note = "PASS", ""
        else:
            status, note = "FAIL", ""
        all_ok = all_ok and status == "PASS"
        print(f"  {label:34s} {fname:24s} {p}/{total}  {status}{note}")
    # any test file not in the curated map -> surface it so nothing hides
    for fname in sorted(set(counts) - listed - {"?"}):
        p, f = counts[fname]
        print(f"  {'[?]  uncategorised':34s} {fname:24s} {p}/{p + f}  "
              f"{'PASS' if f == 0 else 'FAIL'}")
        all_ok = all_ok and f == 0
    if "?" in counts:
        p, f = counts["?"]
        print(f"  {'[?]  unattributed':34s} {'(collection error?)':24s} "
              f"{p}/{p + f}  FAIL")
        all_ok = False
    return all_ok


# ---- hedging block (phase v1) -----------------------------------------------

def hedge_block():
    from sigcore.hedge import embedding as hemb, objective as hobj, solve as hsolve
    from sigcore.hedge import scaling as hscale, pnl as hpnl

    print("[HEDGE] Signature hedger  (single asset, forward & Asian, Q-measure)")
    print("  why: turns the validated primitives into a hedger and shows the "
          "risk-profile trade in numbers.")
    ok = True

    # H1 -- complete-market replication oracle (r=0 forward).
    t, paths = gbm.simulate(1.0, 0.0, 0.2, 1.0, 50, 40_000, seed=0)
    es, _ = hemb.expected_signature(t, paths, 4)
    basis1 = hobj.strategy_basis(1)
    A, b = hobj.mean_variance_system(hemb.forward_covector(1.0, 1.0), 0.0, basis1, es)
    ell1 = hsolve.solve_mean_variance(A, b, lam=0.0)
    tf, pf = gbm.simulate(1.0, 0.0, 0.2, 1.0, 50, 5_000, seed=1)
    Lrep = hpnl.shortfall({(): 1.0}, hpnl.forward_payoff, 0.0, tf, pf, 2, 1.0)
    h1_ok = abs(ell1[0] - 1.0) < 1e-2 and np.max(np.abs(Lrep)) < 1e-9
    ok = ok and h1_ok
    print(f"  H1 replication: ell*_0={ell1[0]:.4f} (target 1)  "
          f"constant-1 max|L|={np.max(np.abs(Lrep)):.1e}   "
          f"{'PASS' if h1_ok else 'FAIL'}")

    # H4 -- scaling reduces cond(A). Use a depth-2 system (the depth-1
    # replication A above is tiny and already well-conditioned).
    td, pd = gbm.simulate(1.0, 0.03, 0.2, 1.0, 60, 60_000, seed=0)
    payoff_f = hemb.forward_covector(1.0, 1.0)
    es2, _ = hemb.expected_signature(td, pd, hobj.mean_variance_required_level(payoff_f, 2))
    A2, _ = hobj.mean_variance_system(payoff_f, 1.0 - 1.0, hobj.strategy_basis(2), es2)
    c_un, c_sc = hscale.condition_numbers(A2)
    print(f"  H4 cond(A) depth-2: unscaled={c_un:.1e}  scaled={c_sc:.1e}  "
          f"({c_un / c_sc:.0f}x)   {'PASS' if c_sc < c_un / 5 else 'FAIL'}")
    ok = ok and c_sc < c_un / 5

    # H3 -- variance reduction by depth under Heston + coarse rebalancing.
    hp = dict(v0=0.04, r=0.0, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7)
    payoff = hemb.asian_covector(1.0, 1.0, 1.0)

    def fit_mv(depth, seed=0, n=20_000):
        tt, pr, _ = heston.simulate(1.0, hp["v0"], hp["r"], hp["kappa"], hp["theta"],
                                    hp["xi"], hp["rho"], 1.0, 60, n, seed=seed)
        e, _ = hemb.expected_signature(tt, pr, hobj.mean_variance_required_level(payoff, depth))
        bs = hobj.strategy_basis(depth)
        Am, bm = hobj.mean_variance_system(payoff, 0.0, bs, e)
        return dict(zip(bs, hsolve.solve_mean_variance(Am, bm, lam=0.0)))

    tt, pr, _ = heston.simulate(1.0, hp["v0"], hp["r"], hp["kappa"], hp["theta"],
                                hp["xi"], hp["rho"], 1.0, 60, 20_000, seed=99)
    var_un = hpnl.shortfall({}, hpnl.asian_payoff, 0.0, tt, pr, 1, 1.0,
                            rebalance_steps=12).var(ddof=1)
    curve = []
    for d in (0, 1, 2):
        L = hpnl.shortfall(fit_mv(d), hpnl.asian_payoff, 0.0, tt, pr, max(d, 1), 1.0,
                           rebalance_steps=12)
        curve.append(L.var(ddof=1))
    h3_ok = curve[0] < var_un and curve[1] <= curve[0] * 1.02 and curve[2] <= curve[1] * 1.02
    ok = ok and h3_ok
    print(f"  H3 Var(L): unhedged={var_un:.2e}  d0={curve[0]:.2e}  d1={curve[1]:.2e}  "
          f"d2={curve[2]:.2e}   {'PASS' if h3_ok else 'FAIL'}")

    # H6 -- asymmetric penalty reshapes the tail (headline table).
    res = _hedge_h6_numbers()
    print("  H6 mean-variance vs asymmetric (Heston, coarse rebalance):")
    print(f"     Var    MV={res['mv']['var']:.3e}  asym={res['as']['var']:.3e}")
    print(f"     CVaR95 MV={res['mv']['cvar95']:.4f}  asym={res['as']['cvar95']:.4f}")
    print(f"     skew   MV={res['mv']['skew']:.3f}  asym={res['as']['skew']:.3f}")
    h6_ok = (res["as"]["var"] >= res["mv"]["var"] * 0.999
             and res["as"]["cvar95"] < res["mv"]["cvar95"]
             and res["as"]["skew"] < res["mv"]["skew"])
    ok = ok and h6_ok
    print(f"     variance-for-tail trade   {'PASS' if h6_ok else 'FAIL'}")
    return ok


def _hedge_h6_numbers():
    # single source of truth: the H6-calibrated computation lives in the test.
    tests_dir = os.path.join(HERE, "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import test_hedge_h6 as h6
    return h6.compute_h6()


def ma_block():
    import iisignature
    from sigcore.hedge import multiasset as ma, solve as hsolve, scaling as hscale
    print("[MULTI] Multi-asset extension  (2-3 assets, block-structured hedge)")
    print("  why: cross-asset coordinates are invisible to single-asset tests; "
          "M0/M-I validate the joint engine first.")
    ok = True
    T = 1.0
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    x0, sig = [1.0, 1.0], [0.2, 0.25]

    # M0 cross coordinate = rho*T
    t, W = gbm.simulate_correlated_brownian_multi(corr, T, 200, 60_000, seed=0)
    emb = np.stack([np.broadcast_to(t, W[0].shape), W[0], W[1]], axis=-1)
    sigs = iisignature.sig(np.ascontiguousarray(emb), 2)
    mc = sigs.mean(0)
    cross = mc[signature.word_index((1, 2), 3)] + mc[signature.word_index((2, 1), 3)]
    m0_ok = abs(cross - corr[0, 1] * T) < 0.02
    print(f"  M0 cross E[S12+S21]={cross:.4f} target {corr[0,1]*T:.4f}   "
          f"{'PASS' if m0_ok else 'FAIL'}")

    # MH1 spread replication -> (+1,-1)
    payoff = ma.spread_forward_covector(x0, 0.0, 0, 1, 2)
    tg, Xg = gbm.simulate_correlated_gbm(x0, 0.0, sig, corr, T, 50, 40_000, seed=0)
    es, _ = ma.expected_signature(tg, Xg, 2)
    A, b, idx = ma.mean_variance_system(payoff, 0.0, ma.strategy_basis(2, 0), 2, es)
    th = hsolve.solve_mean_variance(A, b, 0.0)
    mh1_ok = abs(th[idx[(0, ())]] - 1) < 1e-2 and abs(th[idx[(1, ())]] + 1) < 1e-2
    print(f"  MH1 spread theta=({th[idx[(0,())]]:+.3f},{th[idx[(1,())]]:+.3f}) "
          f"target (+1,-1)   {'PASS' if mh1_ok else 'FAIL'}")

    # MH4c pairs preserved under ridge at rho=0.95
    corr95 = np.array([[1.0, 0.95], [0.95, 1.0]])
    t9, X9 = gbm.simulate_correlated_gbm(x0, 0.0, sig, corr95, T, 50, 40_000, seed=0)
    es9, _ = ma.expected_signature(t9, X9, 2)
    A9, b9, idx9 = ma.mean_variance_system(payoff, 0.0, ma.strategy_basis(2, 0), 2, es9)
    c_un, c_sc = hscale.condition_numbers(A9)
    thr = hscale.ridged_solve(A9, b9, lam=1e-3)
    mh4_ok = abs(thr[idx9[(0, ())]] - 1) < 5e-2 and abs(thr[idx9[(1, ())]] + 1) < 5e-2
    print(f"  MH4 rho=.95 cond(A)={c_un:.1e}; ridged spread="
          f"({thr[idx9[(0,())]]:+.3f},{thr[idx9[(1,())]]:+.3f})   "
          f"{'PASS' if mh4_ok else 'FAIL'}")

    # MH6 asymmetric tail (multi-asset headline)
    tests_dir = os.path.join(HERE, "tests")
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    import test_multiasset_h6 as h6
    r = h6.compute_mh6()
    mh6_ok = (r["as"]["var"] >= r["mv"]["var"] * 0.999
              and r["as"]["cvar95"] < r["mv"]["cvar95"]
              and r["as"]["skew"] < r["mv"]["skew"])
    print(f"  MH6 (correlated, incomplete): CVaR95 MV={r['mv']['cvar95']:.4f} "
          f"asym={r['as']['cvar95']:.4f}  Var {r['mv']['var']:.2e}->{r['as']['var']:.2e}"
          f"   {'PASS' if mh6_ok else 'FAIL'}")
    return ok and m0_ok and mh1_ok and mh4_ok and mh6_ok


def main():
    print("=== Signature Pricing Core — Validation Report ===")
    print(f"params: x0={X0}  r={R}  sigma={SIGMA}  T={T}  n_steps={N_STEPS}  "
          f"n_paths={N_PATHS}  level={LEVEL}  seed={SEED}")
    print()

    results = []
    for check in (t1, t2, t3, t4, b_block, d_block, u_block, i_block, m_block,
                  cond_block, hedge_block):
        results.append(check())
        print()

    # Authoritative enumeration: a single report.py run can no longer hide a
    # missing or failing category (the narrative blocks above are a subset).
    inventory_ok = category_inventory()
    print()

    if all(results) and inventory_ok:
        print("All checks passed. Core is trusted above level 1.")
        return 0
    print("Some checks FAILED. Core is NOT trusted.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
