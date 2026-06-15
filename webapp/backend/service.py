"""Service layer: every demo number comes from a real sigcore computation.

Single-asset focus (2-asset spread is a secondary view). The hedge is fit by the
per-path objective E[P(L)] (= <P^sh(L), E[S]> by the shuffle identity), using the
running-signature feedback theta_t = <ell, S_{0,t}> validated in the hedging
phase. Incompleteness (so the risk profile actually moves the tail) comes from
Heston and/or coarse rebalancing — never fine-grid GBM (a complete market).
"""
from __future__ import annotations

import numpy as np
import iisignature

from sigcore import gbm, heston, signature
from sigcore.hedge import embedding as emb, strategy, multiasset as ma

TIME, LEAD, LAG = emb.TIME, emb.LEAD, emb.LAG
N_CHANNELS = emb.N_CHANNELS


# --- models ------------------------------------------------------------------

def simulate(model, params, n_steps, n_paths, seed):
    """Return (times, paths) of the price under the chosen model."""
    if model == "gbm":
        return gbm.simulate(params.get("x0", 1.0), params.get("r", 0.0),
                            params.get("sigma", 0.2), params.get("T", 1.0),
                            n_steps, n_paths, seed)
    # heston (default — incomplete)
    t, price, _ = heston.simulate(
        params.get("x0", 1.0), params.get("v0", 0.04), params.get("r", 0.0),
        params.get("kappa", 2.0), params.get("theta", 0.04), params.get("xi", 0.5),
        params.get("rho", -0.7), params.get("T", 1.0), n_steps, n_paths, seed)
    return t, price


# --- derivatives (closed-form covectors on the enlarged path) ----------------

_TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def _trapz(price, times):
    return _TRAPZ(price, times, axis=1)


DERIVATIVES = {
    "forward": {
        "label": "Forward",
        "blurb": "The price at maturity.",
        "covector": "(X₀ − K)·∅  +  1·(price)",
        "level": 1,
        "complete": True,
        "payoff": lambda t, X, K: X[:, -1] - K,
    },
    "asian": {
        "label": "Asian (average)",
        "blurb": "The average price over the life, not just the end.",
        "covector": "(X₀ − K)·∅  +  (1/T)·(price, time)",
        "level": 2,
        "complete": False,
        "payoff": lambda t, X, K: _trapz(X, t) / (t[-1] - t[0]) - K,
    },
    "varswap": {
        "label": "Realised-variance swap",
        "blurb": "A bet on how much the price moved, regardless of where it ended.",
        "covector": "−K·∅  +  1·(price↑,price)  −  1·(price,price↑)",
        "level": 2,
        "complete": False,
        "payoff": lambda t, X, K: np.sum(np.diff(X, axis=1) ** 2, axis=1) - K,
    },
}


def price_and_strike(model, params, deriv_key, K=None, n_steps=60, n_paths=40000, seed=1):
    """Strike and fair price on ONE sample (so they are mutually consistent).

    If ``K`` is None the contract is struck **at par** (at-the-money):
    ``K = E[payoff @ K=0]``, which makes the fair value ``p0 = e^{-rT} E[payoff-K]``
    exactly zero (up to nothing — same sample), as for a variance swap struck at
    fair variance or an at-the-money forward. A user-supplied ``K`` gives a
    genuine nonzero price. Returns ``(K, p0, se)``; ``se`` is the price's MC
    standard error (the honest uncertainty in p0)."""
    t, X = simulate(model, params, n_steps, n_paths, seed)
    raw = DERIVATIVES[deriv_key]["payoff"](t, X, 0.0)
    at_par = K is None
    if at_par:
        K = float(raw.mean())
    r, T = params.get("r", 0.0), params.get("T", 1.0)
    disc = np.exp(-r * T)
    payoff = raw - K
    p0 = 0.0 if at_par else float(disc * payoff.mean())
    se = float(disc * payoff.std(ddof=1) / np.sqrt(payoff.size))
    return K, p0, se


def strike_for_par(model, params, deriv_key, n_steps=60, n_paths=40000, seed=1):
    return price_and_strike(model, params, deriv_key, None, n_steps, n_paths, seed)[0]


def fair_price(model, params, deriv_key, K, n_steps=60, n_paths=40000, seed=1):
    _, p0, se = price_and_strike(model, params, deriv_key, K, n_steps, n_paths, seed)
    return p0, se


# --- hedge fitting (per-path objective; any penalty degree, low level) -------

def _strategy_basis(depth):
    import itertools
    letters = (TIME, LAG)
    words = [()]
    for L in range(1, depth + 1):
        words.extend(itertools.product(letters, repeat=L))
    return words


def _trade_matrix(basis, times, paths, rebalance_steps, level):
    """Per path, the P&L of each unit basis strategy: T[i, w] = sum_k <w,S_{0,t_k}> dX_k.

    These are the regressors the hedge is a linear combination of.
    """
    n = times.size
    if rebalance_steps and rebalance_steps < n - 1:
        idx = np.unique(np.linspace(0, n - 1, rebalance_steps + 1).round().astype(int))
    else:
        idx = np.arange(n)
    rt = times[idx]
    npaths = paths.shape[0]
    nb = len(basis)
    out = np.zeros((npaths, nb))
    word_idx = [None if not w else signature.word_index(w, N_CHANNELS) for w in basis]
    for p in range(npaths):
        series = paths[p, idx]
        enl = emb.enlarged(rt, series)
        dX = np.diff(series)
        stream = iisignature.sig(enl, level, 2) if len(series) > 1 else None
        for bi, w in enumerate(basis):
            theta = np.empty(len(series))
            theta[0] = 1.0 if not w else 0.0
            for k in range(1, len(series)):
                theta[k] = 1.0 if not w else stream[2 * k - 1][word_idx[bi]]
            out[p, bi] = np.sum(theta[:-1] * dX)
    return out


def _newton_fit(payoff, trades, p0, penalty, iters=80):
    """Minimise mean(P(L)), L = payoff - p0 - trades @ ell.  penalty=(gamma,delta)."""
    g, d = penalty
    nb = trades.shape[1]
    ell = np.zeros(nb)
    ridge = 1e-8 * np.eye(nb)
    for _ in range(iters):
        L = payoff - p0 - trades @ ell
        Pp = 2 * L + 3 * g * L ** 2 + 4 * d * L ** 3
        Ppp = 2 + 6 * g * L + 12 * d * L ** 2
        grad = -(trades * Pp[:, None]).mean(0)
        hess = np.einsum("i,ij,ik->jk", Ppp, trades, trades) / trades.shape[0] + ridge
        step = np.linalg.solve(hess, -grad)
        ell = ell + step
        if np.linalg.norm(step) < 1e-12:
            break
    return ell


def _stats(L, tau):
    q = float(np.quantile(L, 0.95))
    return {
        "mean": float(L.mean()),
        "variance": float(L.var(ddof=1)),
        "cvar95": float(L[L >= q].mean()),
        "skew": float(np.mean(((L - L.mean()) / L.std()) ** 3)),
        "p_exceed": float(np.mean(L > tau)),
    }


def hedge_and_pnl(model, params, deriv_key, K, p0, gamma, delta, depth=1,
                  rebalance_steps=12, n_steps=60, n_paths=8000, seed=2):
    """Fit the mean-variance and selected-polynomial hedges; return holdings and
    the two shortfall distributions with summary numbers. ``K`` and ``p0`` are
    passed in (computed once, consistently) so a par contract has p0 = 0."""
    deriv = DERIVATIVES[deriv_key]
    basis = _strategy_basis(depth)
    level = max(2 * (depth + 1), 4)

    t, X = simulate(model, params, n_steps, n_paths, seed)
    payoff = deriv["payoff"](t, X, K)
    trades = _trade_matrix(basis, t, X, rebalance_steps, level)
    ell_mv = _newton_fit(payoff, trades, p0, (0.0, 0.0))
    ell_as = _newton_fit(payoff, trades, p0, (gamma, delta))

    L_mv = payoff - p0 - trades @ ell_mv
    L_as = payoff - p0 - trades @ ell_as
    L_un = payoff - p0
    tau = float(np.quantile(L_mv, 0.95))

    # holdings theta_t for a few sample paths (feedback feature)
    sample = min(3, n_paths)
    holdings = []
    for p in range(sample):
        th = strategy.positions(dict(zip(basis, ell_as)), t, X[p], level)
        holdings.append({"t": t.tolist(), "theta": th.tolist(),
                         "price": X[p].tolist()})

    return {
        "p0": p0,
        "ell_mv": dict(zip([str(w) for w in basis], ell_mv.tolist())),
        "ell_as": dict(zip([str(w) for w in basis], ell_as.tolist())),
        "holdings": holdings,
        "L_mv": L_mv.tolist(),
        "L_as": L_as.tolist(),
        "L_unhedged": L_un.tolist(),
        "tau": tau,
        "stats_mv": _stats(L_mv, tau),
        "stats_as": _stats(L_as, tau),
        "stats_unhedged": _stats(L_un, tau),
        "complete": deriv["complete"] and model == "gbm" and rebalance_steps >= n_steps,
    }


def sample_paths(model, params, n_steps=60, n_paths=40, seed=0):
    t, X = simulate(model, params, n_steps, n_paths, seed)
    terminal = X[:, -1]
    return {"t": t.tolist(),
            "paths": X[:min(12, n_paths)].tolist(),
            "terminal": terminal.tolist()}


# --- 3-asset portfolio (the simplex view) ------------------------------------
# A fixed, generated 3-asset example (correlated GBM, coarse rebalancing for
# incompleteness). Hedging a basket liability gives a holdings vector whose
# dollar allocation w^m_t = theta^m_t X^m_t / sum_k theta^k X^k evolves over the
# life of the trade -- traced on the 2-simplex. The risk profile reshapes it.

PORT_X0 = [1.0, 1.0, 1.0]
PORT_SIGMA = [0.20, 0.32, 0.26]
PORT_CORR = [[1.0, 0.55, 0.25], [0.55, 1.0, 0.40], [0.25, 0.40, 1.0]]
PORT_WEIGHTS = [0.40, 0.35, 0.25]            # the basket the portfolio replicates
PORT_LABELS = ["Asset A", "Asset B", "Asset C"]


def _ma_trade_matrix(basis, times, X, rebalance_steps, level, M):
    """Per path, the P&L of each unit (asset, basis-word) strategy: the columns
    the multi-asset hedge is a linear combination of (stacked asset-major)."""
    d = ma.layout(M)["d"]
    n = times.size
    if rebalance_steps and rebalance_steps < n - 1:
        idx = np.unique(np.linspace(0, n - 1, rebalance_steps + 1).round().astype(int))
    else:
        idx = np.arange(n)
    rt = times[idx]
    nb = len(basis)
    npaths = X[0].shape[0]
    widx = [None if not w else signature.word_index(w, d) for w in basis]
    out = np.zeros((npaths, M * nb))
    for p in range(npaths):
        series = [X[m][p, idx] for m in range(M)]
        enl = ma.enlarged(rt, series)
        stream = iisignature.sig(enl, level, 2) if len(rt) > 1 else None
        dX = [np.diff(s) for s in series]
        for bi, w in enumerate(basis):
            wv = np.empty(len(rt))
            wv[0] = 1.0 if not w else 0.0
            for k in range(1, len(rt)):
                wv[k] = 1.0 if not w else stream[2 * k - 1][widx[bi]]
            for m in range(M):
                out[p, m * nb + bi] = np.sum(wv[:-1] * dX[m])
    return out


def _split_assets(ell, basis, M):
    nb = len(basis)
    return [dict(zip(basis, ell[m * nb:(m + 1) * nb])) for m in range(M)]


def _resample_fixed_distance(t, w, ds):
    """Resample a simplex trajectory so consecutive nodes are a FIXED distance
    ``ds`` apart (arc-length reparameterisation, no smoothing -- nodes lie on the
    true path). Carries the time coordinate along so hover stays meaningful."""
    w = np.asarray(w, float)
    seg = np.linalg.norm(np.diff(w, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return [float(t[0]), float(t[-1])], [w[0].tolist(), w[-1].tolist()]
    targets = np.append(np.arange(0.0, total, ds), total)
    rt, rw, j = [], [], 0
    for s in targets:
        while j < len(cum) - 2 and cum[j + 1] < s:
            j += 1
        denom = cum[j + 1] - cum[j]
        f = 0.0 if denom < 1e-12 else (s - cum[j]) / denom
        rw.append((w[j] * (1 - f) + w[j + 1] * f).tolist())
        rt.append(float(t[j] * (1 - f) + t[j + 1] * f))
    return rt, rw


def portfolio(gamma, delta, depth=1, rebalance_steps=4, n_steps=60,
              n_fit=8000, n_show=3, seed=2, T=1.0, r=0.0):
    M = 3
    basis = ma.strategy_basis(M, depth)
    level = 2 * (depth + 1)
    K = float(np.dot(PORT_WEIGHTS, PORT_X0))         # par strike (E[basket] at r=0)

    def basket(X, tt):
        # basket ASIAN: a path-dependent (incomplete) liability, so the risk
        # profile reshapes the optimal allocation, and the depth-1 feedback makes
        # the holdings reallocate dynamically as the assets diverge.
        avg = [_TRAPZ(X[m], tt, axis=1) / T for m in range(M)]
        return sum(PORT_WEIGHTS[m] * avg[m] for m in range(M)) - K

    # fit the mean-variance and downside-averse hedges (per-path objective)
    tf, Xf = gbm.simulate_correlated_gbm(PORT_X0, r, PORT_SIGMA, PORT_CORR, T,
                                         n_steps, n_fit, seed=1)
    trades = _ma_trade_matrix(basis, tf, Xf, rebalance_steps, level, M)
    payoff = basket(Xf, tf)
    ell_mv = _newton_fit(payoff, trades, 0.0, (0.0, 0.0))
    ell_as = _newton_fit(payoff, trades, 0.0, (gamma, delta))

    # representative sample paths: dollar-weight allocation trajectory on the simplex
    ts, Xs = gbm.simulate_correlated_gbm(PORT_X0, r, PORT_SIGMA, PORT_CORR, T,
                                         n_steps, n_show, seed=seed)

    def trajectories(ell, ds=0.005):
        by_asset = _split_assets(ell, basis, M)
        out = []
        for p in range(n_show):
            series = [Xs[m][p] for m in range(M)]
            theta = ma.positions(by_asset, ts, series, level, M)          # (M, n_t)
            gross = np.abs(np.array([theta[m] * series[m] for m in range(M)]))  # exposure
            tot = gross.sum(axis=0)
            n_t = tot.size
            # gross-exposure share; when the Asian hedge unwinds to ~0 exposure
            # near maturity the mix is a 0/0 artifact -> carry the last live
            # allocation forward (degeneracy handling, not smoothing).
            floor = 0.08 * float(tot.max())
            w = np.empty((n_t, M))
            last = np.array(PORT_WEIGHTS, dtype=float)
            for k in range(n_t):
                if tot[k] > floor:
                    last = gross[:, k] / tot[k]
                w[k] = last
            rt, rw = _resample_fixed_distance(ts, w, ds)   # nodes a fixed distance apart
            out.append({"t": rt, "w": rw})
        return out

    return {
        "labels": PORT_LABELS,
        "weights": PORT_WEIGHTS,
        "mv": trajectories(ell_mv),        # each: {t:[...], w:[[wA,wB,wC],...]}
        "asym": trajectories(ell_as),
    }
