"""Dupire local volatility from an implied volatility surface.

Dupire (1994) formula expressed in terms of the Black-Scholes implied
volatility surface sigma(K, T). Using w(K,T) = sigma(K,T)^2 * T and
y = ln(K / F) for forward moneyness, the local variance is

    sigma_loc^2(K, T) = ---------------------------------------------
                            dw/dT
                     -------------------------------------------------
                     1 - (y/w) dw/dy + 0.25*(-0.25 - 1/w + y^2/w^2)(dw/dy)^2
                         + 0.5 * d2w/dy2

This module implements a finite-difference estimator: given a smooth
surface of implied vols (typically SABR-generated), compute the local
volatility at interior grid points.

Reference: Hagan et al. (2002) §2 "Local volatility models" (eq. 2.8),
Dupire (1994) "Pricing with a smile".
"""
from __future__ import annotations

import numpy as np


def dupire_local_vol(iv_surface, strikes, maturities, forward):
    """Compute local volatility on a grid via Dupire's formula.

    Parameters
    ----------
    iv_surface : (nT, nK) array
        Black implied volatility surface; row i = maturity T_i, col j = K_j.
    strikes : (nK,) array
        Strike grid (assumed sorted ascending).
    maturities : (nT,) array
        Maturity grid in years (assumed sorted ascending).
    forward : float
        Forward price F (assumed constant across the grid; acceptable for
        a local sensitivity analysis).

    Returns
    -------
    sigma_loc : (nT, nK) array
        Local volatility at each (T_i, K_j). NaN at boundary rows/columns
        where finite differences are not available.
    """
    iv = np.asarray(iv_surface, dtype=float)
    K = np.asarray(strikes, dtype=float)
    T = np.asarray(maturities, dtype=float)

    nT, nK = iv.shape
    assert K.shape == (nK,)
    assert T.shape == (nT,)

    y = np.log(K / forward)        # log-moneyness
    w = iv ** 2 * T[:, None]       # total variance w(K,T)

    sigma_loc = np.full_like(iv, np.nan)

    for i in range(1, nT - 1):
        dT = T[i + 1] - T[i - 1]
        for j in range(1, nK - 1):
            dwdT = (w[i + 1, j] - w[i - 1, j]) / dT
            dy_plus = y[j + 1] - y[j]
            dy_minus = y[j] - y[j - 1]
            dwdy = (w[i, j + 1] - w[i, j - 1]) / (y[j + 1] - y[j - 1])
            d2wdy2 = 2.0 * (
                (w[i, j + 1] - w[i, j]) / dy_plus
                - (w[i, j] - w[i, j - 1]) / dy_minus
            ) / (dy_plus + dy_minus)

            wij = w[i, j]
            yj = y[j]
            denom = (
                1.0
                - (yj / wij) * dwdy
                + 0.25 * (-0.25 - 1.0 / wij + yj ** 2 / wij ** 2) * dwdy ** 2
                + 0.5 * d2wdy2
            )
            if denom <= 0 or dwdT <= 0:
                continue
            sigma_loc[i, j] = np.sqrt(dwdT / denom)

    return sigma_loc
