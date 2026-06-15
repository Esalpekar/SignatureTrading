"""Solve for the optimal hedge covector.

Mean-variance is a ridged linear solve ``(A + lambda I) ell = b``. The convex
quartic is minimised by damped Newton on the analytic gradient/Hessian;
mean-variance is exactly the one-Newton-step special case (constant Hessian).
"""
from __future__ import annotations

import numpy as np

from . import objective as obj


def solve_mean_variance(A, b, lam=0.0):
    """``(A + lambda I)^{-1} b``."""
    n = A.shape[0]
    return np.linalg.solve(A + lam * np.eye(n), b)


def newton_general(R, grad, hess, x0, lam=0.0, max_iter=50, tol=1e-10):
    """Damped Newton on arbitrary ``R``/``grad``/``hess`` callables (with ridge).

    Used by the multi-asset convex solve, which supplies block gradient/Hessian.
    """
    x = np.array(x0, float)
    n = x.size
    iters = 0
    for iters in range(1, max_iter + 1):
        g = grad(x) + lam * x
        if np.linalg.norm(g) < tol:
            break
        step = np.linalg.solve(hess(x) + lam * np.eye(n), -g)
        t, R0 = 1.0, R(x) + 0.5 * lam * float(x @ x)
        while t > 1e-10 and (R(x + t * step) + 0.5 * lam * float((x + t * step) @ (x + t * step))) > R0 + 1e-4 * t * (g @ step):
            t *= 0.5
        x = x + t * step
        if np.linalg.norm(t * step) < tol:
            break
    return x, {"iters": iters, "grad_norm": float(np.linalg.norm(grad(x) + lam * x))}


def newton(basis, payoff_cov, p0, penalty_coeffs, expected_sig,
           x0=None, lam=0.0, max_iter=50, tol=1e-10):
    """Damped Newton minimisation of ``R(ell)`` for a convex penalty.

    Returns ``(ell, info)``. The step is line-searched (backtracking) so a raw
    full step cannot overshoot far from the optimum, even though convexity
    guarantees eventual convergence. ``lam`` adds a ridge ``lambda I`` to the
    Hessian (and a matching ``lambda*ell`` to the gradient) for conditioning.
    """
    n = len(basis)
    x = np.zeros(n) if x0 is None else np.array(x0, float)

    def R(v):
        loss = obj.build_loss(v, basis, payoff_cov, p0)
        return (obj.objective_value(loss, penalty_coeffs, expected_sig)
                + 0.5 * lam * float(v @ v))

    def grad(v):
        g = obj.gradient(v, basis, payoff_cov, p0, penalty_coeffs, expected_sig)
        return g + lam * v

    def hess(v):
        Hm = obj.hessian(v, basis, payoff_cov, p0, penalty_coeffs, expected_sig)
        return Hm + lam * np.eye(n)

    iters = 0
    for iters in range(1, max_iter + 1):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        step = np.linalg.solve(hess(x), -g)
        # backtracking line search on R
        t, R0, g0 = 1.0, R(x), g
        while t > 1e-10 and R(x + t * step) > R0 + 1e-4 * t * (g0 @ step):
            t *= 0.5
        x = x + t * step
        if np.linalg.norm(t * step) < tol:
            break

    return x, {"iters": iters, "grad_norm": float(np.linalg.norm(grad(x)))}
