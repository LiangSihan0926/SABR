"""SABR implied volatility formula (Hagan, Kumar, Lesniewski, Woodward 2002).

Reference: "Managing Smile Risk", Wilmott Magazine, Sept 2002.
Formulas (2.17a), (2.17b), (2.17c) for F != K and (2.18) for the
at-the-money limit F = K.

Model:
    dF_t     = alpha_t * F_t^beta * dW1
    dalpha_t = nu * alpha_t      * dW2
    dW1 dW2  = rho dt

Parameters
----------
alpha : overall volatility level (alpha > 0)
beta  : elasticity / backbone exponent, beta in [0, 1]
rho   : correlation between F and vol, rho in (-1, 1)
nu    : vol-of-vol, nu > 0
"""
from __future__ import annotations

import numpy as np


def _atm_term(F, T, alpha, beta, rho, nu):
    """ATM (F=K) implied vol, Hagan eq. (2.18)."""
    F = np.asarray(F, dtype=float)
    one_m_b = 1.0 - beta
    Fb = F ** one_m_b

    term1 = (one_m_b ** 2) * alpha ** 2 / (24.0 * F ** (2.0 * one_m_b))
    term2 = 0.25 * rho * beta * nu * alpha / Fb
    term3 = (2.0 - 3.0 * rho ** 2) * nu ** 2 / 24.0
    correction = 1.0 + (term1 + term2 + term3) * T

    return (alpha / Fb) * correction


def sabr_vol_atm(F, T, alpha, beta, rho, nu):
    """Public ATM helper (Hagan 2.18)."""
    return _atm_term(F, T, alpha, beta, rho, nu)


def sabr_vol(K, F, T, alpha, beta, rho, nu, atm_tol=1e-12):
    """SABR lognormal (Black) implied vol, Hagan formula (2.17a-c).

    Inputs
    ------
    K     : strike (scalar or array)
    F     : forward (scalar)
    T     : time to expiry, years (scalar)
    alpha : SABR alpha (> 0)
    beta  : SABR beta (in [0, 1])
    rho   : SABR rho (in (-1, 1))
    nu    : SABR nu (> 0)

    Returns
    -------
    sigma_B : Black implied volatility (same shape as K)

    Numerically stable at the ATM point K == F via the (2.18) ATM limit.
    Near ATM we blend to keep the z / x(z) factor well-defined.
    """
    K = np.asarray(K, dtype=float)
    F_scalar = float(F)
    one_m_b = 1.0 - beta

    out = np.empty_like(K, dtype=float)

    logFK = np.log(F_scalar / K)
    FK_half = (F_scalar * K) ** (one_m_b / 2.0)

    # Denominator from (2.17a): expansion in log(F/K)
    denom = FK_half * (
        1.0
        + (one_m_b ** 2) / 24.0 * logFK ** 2
        + (one_m_b ** 4) / 1920.0 * logFK ** 4
    )

    # z and x(z) from (2.17b), (2.17c)
    z = (nu / alpha) * FK_half * logFK
    # x(z) = log( (sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho) )
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z ** 2)
    xz_numer = sqrt_term + z - rho
    xz = np.log(xz_numer / (1.0 - rho))

    # correction factor {1 + [...] T}
    correction = 1.0 + (
        (one_m_b ** 2) / 24.0 * alpha ** 2 / ((F_scalar * K) ** one_m_b)
        + 0.25 * rho * beta * nu * alpha / FK_half
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    ) * T

    # Main formula (2.17a): sigma = alpha/denom * (z/x(z)) * correction
    # z/x(z) -> 1 as z -> 0; use limit at ATM.
    atm_mask = np.abs(logFK) < atm_tol
    z_over_xz = np.where(atm_mask, 1.0, z / np.where(atm_mask, 1.0, xz))

    out = (alpha / denom) * z_over_xz * correction

    # At exactly ATM use the closed-form (2.18) for numerical stability
    if np.any(atm_mask):
        atm_val = _atm_term(F_scalar, T, alpha, beta, rho, nu)
        out = np.where(atm_mask, atm_val, out)

    return out
