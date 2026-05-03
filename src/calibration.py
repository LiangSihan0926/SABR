"""SABR calibration: fit (alpha, rho, nu) to a market smile by
nonlinear least squares, holding beta fixed (market convention).

Objective
---------
Given observed strikes K_i with market implied vols sigma^mkt_i and
market state (F, T), minimize

    sum_i   w_i * [ sigma_SABR(K_i, F, T; alpha, beta, rho, nu) - sigma^mkt_i ]^2

over (alpha, rho, nu) with bounds
    alpha in [1e-8, inf),  rho in (-0.999, 0.999),  nu in [1e-4, 5.0].

The default initial guess uses the ATM approximation
    alpha0 = sigma_ATM * F^(1-beta)     (first-order from Hagan 2.18)
with rho0 = -0.3, nu0 = 0.4.

The trust-region-reflective solver from SciPy handles the bounds.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import least_squares

from .sabr import sabr_vol


@dataclass
class CalibrationResult:
    alpha: float
    beta: float
    rho: float
    nu: float
    rmse: float              # root-mean-squared IV residual
    max_abs_err: float       # worst single-strike IV residual
    n_points: int            # number of calibration points
    success: bool
    nfev: int                # function evaluations
    message: str

    def as_dict(self):
        return asdict(self)


def calibrate_sabr(
    strikes,
    sigma_market,
    F,
    T,
    beta,
    weights=None,
    x0=None,
    rho_bounds=(-0.999, 0.999),
    nu_bounds=(1e-4, 5.0),
    alpha_lb=1e-8,
) -> CalibrationResult:
    """Calibrate SABR (alpha, rho, nu) to one smile.

    Parameters
    ----------
    strikes      : (n,) array of strikes
    sigma_market : (n,) array of observed implied vols (decimal)
    F, T         : forward and time-to-expiry (years)
    beta         : held fixed in [0, 1]
    weights      : (n,) array of positive weights (default ones)
    x0           : optional (alpha, rho, nu) initial guess
    """
    K = np.asarray(strikes, dtype=float)
    s = np.asarray(sigma_market, dtype=float)
    if K.shape != s.shape or K.ndim != 1:
        raise ValueError("strikes and sigma_market must be 1-D, same length")
    if weights is None:
        w = np.ones_like(K)
    else:
        w = np.asarray(weights, dtype=float)

    if x0 is None:
        # leading-order ATM seed: alpha ≈ sigma_ATM * F^(1-beta)
        # use the market IV closest to ATM as a proxy for sigma_ATM
        i_atm = int(np.argmin(np.abs(K - F)))
        alpha0 = float(s[i_atm]) * F ** (1.0 - beta)
        x0 = np.array([alpha0, -0.30, 0.40])
    else:
        x0 = np.asarray(x0, dtype=float)

    lb = np.array([alpha_lb, rho_bounds[0], nu_bounds[0]])
    ub = np.array([np.inf,   rho_bounds[1], nu_bounds[1]])
    # keep the initial guess strictly interior to the bounds
    x0 = np.clip(x0, lb + 1e-10, np.where(np.isfinite(ub), ub - 1e-10, x0))

    def residuals(x):
        alpha, rho, nu = x
        sigma_model = sabr_vol(K, F, T, alpha, beta, rho, nu)
        return w * (sigma_model - s)

    res = least_squares(
        residuals, x0, bounds=(lb, ub), method="trf",
        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=400,
    )

    err = residuals(res.x) / w          # un-weighted IV residual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_err = float(np.max(np.abs(err)))

    return CalibrationResult(
        alpha=float(res.x[0]),
        beta=float(beta),
        rho=float(res.x[1]),
        nu=float(res.x[2]),
        rmse=rmse,
        max_abs_err=max_err,
        n_points=len(K),
        success=bool(res.success),
        nfev=int(res.nfev),
        message=str(res.message),
    )


def calibrate_smile_panel(smile_df, beta):
    """Convenience wrapper for the output of `data_loader.build_smile`.

    Reads F, T, r from `smile_df.attrs` and strikes / IVs from the frame.
    """
    F = float(smile_df.attrs["F"])
    T = float(smile_df.attrs["expire_dte"]) / 365.25
    K = smile_df["STRIKE"].to_numpy()
    s = smile_df["SIGMA_MKT"].to_numpy()
    return calibrate_sabr(K, s, F, T, beta=beta)
