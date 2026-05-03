"""Unit tests for the core computational primitives.

Run from the repository root with either:

    pytest tests/
    python -m unittest tests.test_core

The four tests cover the analytical identities that motivated the
implementation:

    1. Black 1976 put-call parity                      ->  machine eps
    2. Black implied-volatility round-trip             ->  < 1e-8
    3. SABR (2.17) at K=F  vs  closed-form ATM (2.18)  ->  machine eps
    4. Synthetic SABR parameter recovery               ->  < 1e-10

All tests are deterministic and run in well under one second.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

# Allow `python tests/test_core.py` from the project root
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src.black import (  # noqa: E402
    black_call,
    black_put,
    implied_vol_from_price,
)
from src.sabr import sabr_vol, sabr_vol_atm  # noqa: E402
from src.calibration import calibrate_sabr  # noqa: E402


class TestBlackPutCallParity(unittest.TestCase):
    """C - P = D(F - K) for European-style options on a forward."""

    def test_parity_across_strikes(self):
        F, T, r, sigma = 100.0, 1.0, 0.05, 0.25
        D = np.exp(-r * T)
        for K in np.linspace(70, 130, 21):
            C = float(black_call(F, K, sigma, T, r))
            P = float(black_put(F, K, sigma, T, r))
            lhs = C - P
            rhs = D * (F - K)
            self.assertLess(
                abs(lhs - rhs),
                1e-12,
                msg=f"parity violated at K={K}: C-P={lhs}, D(F-K)={rhs}",
            )


class TestBlackIVRoundTrip(unittest.TestCase):
    """price = Black(sigma)  ->  invert  ->  recover sigma to <1e-8."""

    def test_round_trip_typical_vols(self):
        F, K, T, r = 100.0, 100.0, 1.0, 0.05
        for sigma in (0.05, 0.10, 0.20, 0.30, 0.50, 0.80):
            price = float(black_call(F, K, sigma, T, r))
            sigma_hat = implied_vol_from_price(
                price, F, K, T, r, "call"
            )
            self.assertLess(
                abs(sigma - sigma_hat),
                1e-8,
                msg=f"round-trip failed for sigma={sigma}",
            )

    def test_round_trip_off_atm(self):
        F, T, r, sigma = 100.0, 0.5, 0.03, 0.30
        for K in (80, 90, 95, 100, 105, 110, 120):
            for typ in ("call", "put"):
                if typ == "call":
                    price = float(black_call(F, K, sigma, T, r))
                else:
                    price = float(black_put(F, K, sigma, T, r))
                sigma_hat = implied_vol_from_price(price, F, K, T, r, typ)
                self.assertLess(
                    abs(sigma - sigma_hat),
                    1e-8,
                    msg=f"round-trip failed for K={K}, type={typ}",
                )


class TestSABRATMConsistency(unittest.TestCase):
    """The general (2.17) formula at K=F must equal the closed-form ATM
    limit (2.18) to machine precision."""

    def test_atm_consistency(self):
        F, T = 100.0, 1.0
        cases = [
            # (alpha, beta, rho, nu)
            (0.20, 0.0, -0.30, 0.40),
            (0.20, 0.5, -0.30, 0.40),
            (0.20, 1.0, -0.30, 0.40),
            (0.30, 0.5, +0.10, 0.20),
            (0.10, 0.5, -0.70, 0.80),
        ]
        for alpha, beta, rho, nu in cases:
            v17 = float(sabr_vol(np.array([F]), F, T,
                                 alpha, beta, rho, nu)[0])
            v18 = sabr_vol_atm(F, T, alpha, beta, rho, nu)
            self.assertLess(
                abs(v17 - v18),
                1e-14,
                msg=(f"ATM consistency failed for "
                     f"(alpha={alpha}, beta={beta}, rho={rho}, nu={nu}): "
                     f"v17={v17}, v18={v18}"),
            )


class TestSyntheticSABRRecovery(unittest.TestCase):
    """Generate a noiseless smile from known SABR parameters,
    re-calibrate, recover the parameters to <1e-10."""

    def test_recovery_noiseless(self):
        F, T = 100.0, 1.0
        alpha_t, beta_t, rho_t, nu_t = 0.20, 0.5, -0.30, 0.40
        K = np.linspace(75, 125, 25)
        sigma = sabr_vol(K, F, T, alpha_t, beta_t, rho_t, nu_t)

        result = calibrate_sabr(K, sigma, F, T, beta=beta_t)

        self.assertTrue(result.success)
        self.assertLess(abs(result.alpha - alpha_t), 1e-10)
        self.assertLess(abs(result.rho - rho_t), 1e-10)
        self.assertLess(abs(result.nu - nu_t), 1e-10)
        self.assertLess(result.rmse, 1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
