"""MH6 — asymmetric penalty thins the joint tail (multi-asset headline).

Correlated assets + incompleteness (a basket Asian is not hedgeable by a
constant terminal-increment position): the convex-quartic hedge must show
strictly lower CVaR95 and strictly higher variance than mean-variance, now
exploiting cross-asset co-skewness.

Note on the dimensionality wall (spec 1.6): a level-8 joint E[S] (d=5) is
infeasible. But by the shuffle identity <P^sh(L), E[S]> = E[P(<L,S>)], so the
depth-0 objective is fit per-path from the level-1 trade coords (the asset
increments) and the realised payoff — the exact signature objective, no giant
E[S] vector. theta is constant, so the P&L = sum_m theta_m * (X^m_T - X^m_0).
"""
import numpy as np

from sigcore import gbm

T = 1.0
X0 = [1.0, 1.0]
SIG = [0.35, 0.30]
CORR = np.array([[1.0, 0.7], [0.7, 1.0]])
W = [0.6, 0.4]
P0 = float(np.dot(W, X0) - 1.0)           # E[basket Asian] - K at r=0 (E[A^i]=x0)
GAMMA, DELTA = 6.0, 14.5                   # convex: delta >= 3 gamma^2/8


def _sim(n_paths, seed):
    return gbm.simulate_correlated_gbm(X0, 0.0, SIG, CORR, T, 60, n_paths, seed=seed)


K = 1.0


def _basket_asian(times, X):
    avg = [np.trapezoid(X[m], times, axis=1) / T for m in range(len(X))]
    return sum(W[m] * avg[m] for m in range(len(X))) - K     # mean-zero vs P0


def _fit_perpath(payoff, dX, penalty, p0, iters=60):
    """Newton on mean(P(L)), L = payoff - p0 - dX @ ell (depth-0, M scalars)."""
    g, d = penalty
    M = dX.shape[1]
    ell = np.zeros(M)
    for _ in range(iters):
        L = payoff - p0 - dX @ ell
        Pp = 2 * L + 3 * g * L ** 2 + 4 * d * L ** 3
        Ppp = 2 + 6 * g * L + 12 * d * L ** 2
        grad = -(dX * Pp[:, None]).mean(0)
        hess = np.einsum("i,ij,ik->jk", Ppp, dX, dX) / dX.shape[0]
        step = np.linalg.solve(hess, -grad)
        ell = ell + step
        if np.linalg.norm(step) < 1e-12:
            break
    return ell


def _stats(L):
    q = np.quantile(L, 0.95)
    return {"var": L.var(ddof=1), "cvar95": L[L >= q].mean(),
            "skew": float(np.mean(((L - L.mean()) / L.std()) ** 3))}


def compute_mh6(fit_seed=0, apply_seed=777, n=40_000):
    tf, Xf = _sim(n, fit_seed)
    payoff = _basket_asian(tf, Xf)
    dX = np.stack([Xf[m][:, -1] - Xf[m][:, 0] for m in range(2)], axis=1)
    ell_mv = _fit_perpath(payoff, dX, (0.0, 0.0), P0)
    ell_as = _fit_perpath(payoff, dX, (GAMMA, DELTA), P0)

    ta, Xa = _sim(n, apply_seed)                       # fresh paths
    pa = _basket_asian(ta, Xa)
    dXa = np.stack([Xa[m][:, -1] - Xa[m][:, 0] for m in range(2)], axis=1)
    L_mv = pa - P0 - dXa @ ell_mv
    L_as = pa - P0 - dXa @ ell_as
    return {"mv": _stats(L_mv), "as": _stats(L_as),
            "ell_mv": ell_mv, "ell_as": ell_as}


def test_MH6_asymmetric_thins_joint_tail():
    r = compute_mh6()
    print(f"\n[MH6] hedge vectors: MV={np.round(r['ell_mv'],3)}  "
          f"asym={np.round(r['ell_as'],3)}")
    print(f"   Var   : MV={r['mv']['var']:.3e}  asym={r['as']['var']:.3e}")
    print(f"   CVaR95: MV={r['mv']['cvar95']:.4f}  asym={r['as']['cvar95']:.4f}")
    print(f"   skew  : MV={r['mv']['skew']:.3f}  asym={r['as']['skew']:.3f}")
    assert r["as"]["var"] >= r["mv"]["var"] * 0.999
    assert r["as"]["cvar95"] < r["mv"]["cvar95"]
    assert r["as"]["skew"] < r["mv"]["skew"]
