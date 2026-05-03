"""Quantitative smile descriptors used for parameter-sensitivity plots.

Given a SABR parameter set (alpha, beta, rho, nu) and a market state
(F, T), we summarize the resulting implied-vol smile by three scalars:

    level    : sigma_SABR(F, F, T)                    (ATM vol)
    slope    : d sigma / d ln K   |_{K=F}             (25d skew proxy)
    curvature: d^2 sigma / d(ln K)^2  |_{K=F}         (smile convexity)

Plus a backbone utility: ATM vol as a function of the forward F, which
is what Hagan figures 2.2–2.4 / eq. (2.15) describe.
"""
from __future__ import annotations

import numpy as np

from .sabr import sabr_vol, sabr_vol_atm


def atm_level(F, T, alpha, beta, rho, nu):
    """ATM vol sigma(F, F, T)."""
    return sabr_vol_atm(F, T, alpha, beta, rho, nu)


def atm_slope(F, T, alpha, beta, rho, nu, h=1e-3):
    """Central finite difference of sigma w.r.t. ln K, evaluated at K=F."""
    Ks = F * np.exp(np.array([-h, h]))
    s = sabr_vol(Ks, F, T, alpha, beta, rho, nu)
    return float((s[1] - s[0]) / (2.0 * h))


def atm_curvature(F, T, alpha, beta, rho, nu, h=1e-2):
    """Central finite difference of sigma w.r.t. ln K (2nd derivative) at K=F.

    Larger h than slope to avoid round-off from subtracting near-equal values.
    """
    Ks = F * np.exp(np.array([-h, 0.0, h]))
    s = sabr_vol(Ks, F, T, alpha, beta, rho, nu)
    return float((s[2] - 2.0 * s[1] + s[0]) / h ** 2)


def backbone(F_grid, T, alpha, beta, rho, nu):
    """ATM vol traced as F moves over F_grid. Hagan eq. (2.15) / fig. 2.2."""
    F_grid = np.asarray(F_grid, dtype=float)
    return np.array([sabr_vol_atm(F, T, alpha, beta, rho, nu) for F in F_grid])


def alpha_for_target_atm(F, T, beta, rho, nu, target_atm, tol=1e-10,
                         max_iter=100):
    """Solve for alpha so that sigma_ATM(F, T; alpha, beta, rho, nu) = target.

    Newton iteration on log alpha; first-order initial guess
    alpha0 = target * F^(1-beta) is already close.
    """
    a = target_atm * F ** (1.0 - beta)       # leading-order guess
    for _ in range(max_iter):
        v = sabr_vol_atm(F, T, a, beta, rho, nu)
        diff = v - target_atm
        if abs(diff) < tol:
            return a
        # numerical derivative d vol_ATM / d alpha
        da = 1e-6 * a
        vp = sabr_vol_atm(F, T, a + da, beta, rho, nu)
        dvda = (vp - v) / da
        if dvda == 0:
            break
        a -= diff / dvda
        a = max(a, 1e-10)
    return a
