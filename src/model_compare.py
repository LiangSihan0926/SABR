"""Helpers for the Module 5 model-comparison study.

Three things we need that the earlier modules do not already provide:

1. A flat Black-Scholes fit to a smile (the BS baseline — one scalar sigma).
2. Prediction of tomorrow's smile under four competing dynamic models,
   given today's model/fit and a new forward F_new:
       (a) **sticky-strike BS**: sigma_new(K) = sigma_today(K)
       (b) **sticky-moneyness BS**: sigma_new(K) = sigma_today(K * F_today/F_new)
       (c) **frozen local vol**: sigma_new(K, T) ≈ sigma_loc(K, T)
           (local vol is a function of absolute K, independent of F)
       (d) **frozen SABR**: sigma_new(K) = sigma_SABR(K, F_new; alpha, beta, rho, nu)
3. A convenience routine for extracting a 1-D local-vol slice at a fixed T
   from a SABR-generated implied-vol surface.

Reference: Hagan, Kumar, Lesniewski, Woodward (2002) §2 — their
figures 2.2–2.4 are exactly the dynamics comparison this module enables.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

from .sabr import sabr_vol
from .local_vol import dupire_local_vol


# ---------------------------------------------------------------
# 1. Flat Black-Scholes baseline
# ---------------------------------------------------------------
def fit_flat_bs(K, sigma_market):
    """Single-sigma Black-Scholes "fit": minimize RMSE over the smile."""
    K = np.asarray(K, dtype=float)
    s = np.asarray(sigma_market, dtype=float)

    def obj(sigma):
        return np.mean((sigma - s) ** 2)

    res = minimize_scalar(obj, bounds=(1e-4, 5.0), method="bounded",
                          options={"xatol": 1e-10})
    sigma_bs = float(res.x)
    rmse = float(np.sqrt(obj(sigma_bs)))
    return sigma_bs, rmse


# ---------------------------------------------------------------
# 2. Local-vol slice at a fixed T from a SABR smile
# ---------------------------------------------------------------
def local_vol_slice(K_grid, F, T, alpha, beta, rho, nu, T_pad=(0.05, 0.05)):
    """Compute sigma_loc(K, T) along a 1-D strike grid.

    Builds a tiny 3-row implied-vol surface at T - T_pad[0], T, T + T_pad[1]
    from SABR, then applies `dupire_local_vol` and returns the middle row.
    """
    T_lo, T_hi = max(1e-6, T - T_pad[0]), T + T_pad[1]
    T_row = np.array([T_lo, T, T_hi])
    iv = np.array([
        sabr_vol(K_grid, F, t, alpha, beta, rho, nu) for t in T_row
    ])
    sloc_surf = dupire_local_vol(iv, K_grid, T_row, F)
    return sloc_surf[1]


# ---------------------------------------------------------------
# 3. Smile prediction under each dynamic model
# ---------------------------------------------------------------
def predict_sticky_strike(K_new, K_today, sigma_today):
    """sigma_new(K) = sigma_today(K)  (pure Black-Scholes strike-space)."""
    interp = interp1d(K_today, sigma_today, kind="linear",
                      bounds_error=False, fill_value=np.nan)
    return interp(K_new)


def predict_sticky_moneyness(K_new, F_new, K_today, F_today, sigma_today):
    """sigma_new(K) = sigma_today(K * F_today/F_new)  (smile translates with F)."""
    interp = interp1d(K_today, sigma_today, kind="linear",
                      bounds_error=False, fill_value=np.nan)
    return interp(K_new * F_today / F_new)


def predict_local_vol(K_new, F_new, T, sigma_loc, K_lv_grid):
    """Approximate prediction under frozen local vol.

    The Hagan §2 / Derman–Kani approximation says that if today's smile
    were produced by local volatility, then tomorrow's implied-vol smile
    is *approximately* given by
         sigma_new(K) ≈ sigma_loc( (K + F_new) / 2, T ).
    This captures the essential (wrong-direction) dynamics and matches
    the paper's figures 2.3–2.4 qualitatively.
    """
    interp = interp1d(K_lv_grid, sigma_loc, kind="linear",
                      bounds_error=False, fill_value=np.nan)
    return interp(0.5 * (K_new + F_new))


def predict_sabr(K_new, F_new, T, alpha, beta, rho, nu):
    """Forecast smile by re-evaluating SABR at (K_new, F_new) with the
    calibration-date parameters (alpha, beta, rho, nu) held fixed."""
    return sabr_vol(K_new, F_new, T, alpha, beta, rho, nu)


# ---------------------------------------------------------------
# Tiny helper: RMSE on overlapping strikes (ignoring NaNs)
# ---------------------------------------------------------------
def rmse(predicted, observed):
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    mask = np.isfinite(predicted) & np.isfinite(observed)
    if mask.sum() == 0:
        return np.nan
    diff = predicted[mask] - observed[mask]
    return float(np.sqrt(np.mean(diff ** 2)))
