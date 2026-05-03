"""Black (1976) formula for options on a forward + implied volatility inverter.

Reference: Hagan et al. (2002), equation (2.4).
    Call: V_call = D(0,T) * [ F * N(d1) - K * N(d2) ]
    Put : V_put  = D(0,T) * [ K * N(-d2) - F * N(-d1) ]
with
    d1,2 = [ ln(F/K) +/- 0.5 * sigma^2 * T ] / (sigma * sqrt(T))
    D(0,T) = exp(-r T)   (discount factor)

Here F is the forward price of the underlying to expiry T,
K is strike, sigma is the Black (log-normal) volatility, r is the
risk-free rate, T is time to expiry in years.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def _d1_d2(F, K, sigma, T):
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def black_call(F, K, sigma, T, r=0.0):
    """Black (1976) call price."""
    d1, d2 = _d1_d2(F, K, sigma, T)
    disc = np.exp(-r * T)
    return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black_put(F, K, sigma, T, r=0.0):
    """Black (1976) put price."""
    d1, d2 = _d1_d2(F, K, sigma, T)
    disc = np.exp(-r * T)
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black_price(F, K, sigma, T, r=0.0, option_type="call"):
    """Dispatch helper: option_type in {'call','put','c','p'}."""
    t = option_type.lower()
    if t in ("call", "c"):
        return black_call(F, K, sigma, T, r)
    if t in ("put", "p"):
        return black_put(F, K, sigma, T, r)
    raise ValueError(f"unknown option_type: {option_type}")


def black_vega(F, K, sigma, T, r=0.0):
    """Vega: dV/dsigma. Same for call and put."""
    d1, _ = _d1_d2(F, K, sigma, T)
    disc = np.exp(-r * T)
    return disc * F * norm.pdf(d1) * np.sqrt(T)


def implied_vol_from_price(price, F, K, T, r=0.0, option_type="call",
                           lo=1e-6, hi=5.0):
    """Invert Black's formula for implied volatility via Brent's method.

    Returns np.nan if the target price is outside arbitrage-free bounds.
    """
    disc = np.exp(-r * T)
    if option_type.lower() in ("call", "c"):
        intrinsic = disc * max(F - K, 0.0)
        upper = disc * F
    else:
        intrinsic = disc * max(K - F, 0.0)
        upper = disc * K
    if price < intrinsic - 1e-12 or price > upper + 1e-12:
        return np.nan

    def objective(sigma):
        return black_price(F, K, sigma, T, r, option_type) - price

    try:
        return brentq(objective, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError:
        return np.nan
