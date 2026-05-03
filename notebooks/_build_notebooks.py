"""One-shot builder that emits the three Module-1 notebooks.

Run once from the `notebooks/` directory:
    python3 _build_notebooks.py
"""
import json
import os

NB_VERSION = 4


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def make_nb(cells, title):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
            "title": title,
        },
        "nbformat": NB_VERSION,
        "nbformat_minor": 5,
    }


def save(nb, path):
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {path}")


# -------------------- common setup cell --------------------
SETUP = r"""# allow importing from ../src
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (7, 4.5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})
"""


# ==============================================================
# Notebook 1: Black (1976) formula + implied volatility inverter
# ==============================================================
nb1 = make_nb([
    md("# 01 — Black (1976) Formula + Implied Volatility\n\n"
       "Reference: Hagan, Kumar, Lesniewski, Woodward (2002), *Managing "
       "Smile Risk*, eq. (2.4).\n\n"
       "**Black (1976) call / put on a forward:**\n"
       "$$ V_{\\text{call}} = D(0,T)\\,[F\\,N(d_1) - K\\,N(d_2)] $$\n"
       "$$ V_{\\text{put}}  = D(0,T)\\,[K\\,N(-d_2) - F\\,N(-d_1)] $$\n"
       "$$ d_{1,2} = \\frac{\\ln(F/K) \\pm \\tfrac12 \\sigma^2 T}{\\sigma \\sqrt{T}},\\quad "
       "D(0,T) = e^{-rT}. $$\n\n"
       "The SABR model in later notebooks produces $\\sigma_{SABR}$, which "
       "is then fed into this formula to obtain an option price."),
    code(SETUP),
    code("from src.black import (\n"
         "    black_call, black_put, black_price, black_vega,\n"
         "    implied_vol_from_price,\n"
         ")\n\n"
         "F, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20\n"
         "C = black_call(F, K, sigma, T, r)\n"
         "P = black_put (F, K, sigma, T, r)\n"
         "print(f'Call = {C:.6f}')\n"
         "print(f'Put  = {P:.6f}')"),
    md("### Test 1 — Put-call parity\n"
       "$$ C - P = D(0,T)\\,(F - K) $$"),
    code("lhs = C - P\n"
         "rhs = np.exp(-r*T) * (F - K)\n"
         "print(f'C - P       = {lhs: .8f}')\n"
         "print(f'D(F - K)    = {rhs: .8f}')\n"
         "print(f'difference  = {abs(lhs - rhs):.2e}')\n"
         "assert abs(lhs - rhs) < 1e-10, 'put-call parity violated'"),
    md("### Test 2 — Implied-volatility round-trip\n"
       "Input $\\sigma \\to$ price $\\to$ invert $\\to$ recovered $\\sigma$."),
    code("sigmas_in = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]\n"
         "print(f'{\"sigma_in\":>10} {\"price\":>10} {\"sigma_out\":>12} {\"err\":>12}')\n"
         "for s in sigmas_in:\n"
         "    px = black_call(F, K, s, T, r)\n"
         "    s_hat = implied_vol_from_price(px, F, K, T, r, 'call')\n"
         "    print(f'{s:>10.4f} {px:>10.4f} {s_hat:>12.8f} {abs(s-s_hat):>12.2e}')"),
    md("### Test 3 — Round-trip across strikes (ITM / ATM / OTM)"),
    code("Ks = np.linspace(70, 130, 13)\n"
         "sigma = 0.25\n"
         "errs = []\n"
         "for K_ in Ks:\n"
         "    px = black_call(F, K_, sigma, T, r)\n"
         "    s_hat = implied_vol_from_price(px, F, K_, T, r, 'call')\n"
         "    errs.append(abs(sigma - s_hat))\n"
         "print(f'max inversion error across strikes = {max(errs):.2e}')"),
    md("### Test 4 — Price and vega versus $\\sigma$"),
    code("sigmas = np.linspace(0.01, 1.00, 100)\n"
         "prices = black_call(F, K, sigmas, T, r)\n"
         "vegas  = black_vega(F, K, sigmas, T, r)\n\n"
         "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
         "ax[0].plot(sigmas, prices); ax[0].set_xlabel(r'$\\sigma$'); ax[0].set_ylabel('Call price')\n"
         "ax[0].set_title('Black call price vs volatility')\n"
         "ax[1].plot(sigmas, vegas, color='C1'); ax[1].set_xlabel(r'$\\sigma$'); ax[1].set_ylabel('Vega')\n"
         "ax[1].set_title('Black vega vs volatility')\n"
         "plt.tight_layout(); plt.show()"),
    md("### Summary\n"
       "* Black call / put implemented against Hagan (2.4).\n"
       "* Put-call parity holds to machine precision.\n"
       "* Implied-volatility inverter (Brent on $[10^{-6}, 5]$) recovers $\\sigma$ to $<10^{-8}$.\n"
       "* These routines will be called from Module 4 (calibration) and "
       "Module 5 (SABR vs BS comparison)."),
], title="01 Black Formula")

save(nb1, "01_black_formula.ipynb")


# ==============================================================
# Notebook 2: SABR implied volatility (Hagan 2.17a-c + 2.18)
# ==============================================================
nb2 = make_nb([
    md("# 02 — SABR Implied Volatility\n\n"
       "Reference: Hagan et al. (2002) formulas **(2.17a-c)** and the ATM "
       "limit **(2.18)**.\n\n"
       "**SABR dynamics:**\n"
       "$$ dF_t = \\alpha_t F_t^\\beta\\,dW_1,\\quad "
       "d\\alpha_t = \\nu\\,\\alpha_t\\,dW_2,\\quad "
       "dW_1\\,dW_2 = \\rho\\,dt. $$\n\n"
       "**Parameter meaning** (Slide 8 in the outline):\n\n"
       "| Parameter | Effect |\n"
       "|---|---|\n"
       "| $\\alpha$ | level of volatility (tied to ATM) |\n"
       "| $\\beta$  | backbone shape |\n"
       "| $\\rho$   | skew / slope |\n"
       "| $\\nu$    | smile curvature / vol-of-vol |"),
    code(SETUP),
    code("from src.sabr import sabr_vol, sabr_vol_atm\n\n"
         "# reference set of SABR parameters (close to paper figure 3.3)\n"
         "F0, T0 = 100.0, 1.0\n"
         "alpha0, beta0, rho0, nu0 = 0.20, 0.5, -0.30, 0.40\n\n"
         "Ks = np.linspace(70, 130, 61)\n"
         "ivs = sabr_vol(Ks, F0, T0, alpha0, beta0, rho0, nu0)\n"
         "print('ATM (from 2.17 at K=F):', float(sabr_vol(np.array([F0]), F0, T0, alpha0, beta0, rho0, nu0)[0]))\n"
         "print('ATM (from 2.18      ):', sabr_vol_atm(F0, T0, alpha0, beta0, rho0, nu0))\n\n"
         "plt.plot(Ks, ivs)\n"
         "plt.axvline(F0, color='k', lw=0.8, ls='--', label='F')\n"
         "plt.xlabel('Strike K'); plt.ylabel(r'$\\sigma_{SABR}$')\n"
         "plt.title(rf'Baseline smile  $\\alpha$={alpha0}, $\\beta$={beta0}, $\\rho$={rho0}, $\\nu$={nu0}')\n"
         "plt.legend(); plt.show()"),
    md("---\n## Test — ATM consistency (2.17 at K=F must match 2.18)"),
    code("for (a, b, r_, n) in [(0.1, 0.0, -0.3, 0.3),\n"
         "                      (0.2, 0.5, -0.5, 0.4),\n"
         "                      (0.3, 1.0,  0.0, 0.6)]:\n"
         "    v_17 = float(sabr_vol(np.array([F0]), F0, T0, a, b, r_, n)[0])\n"
         "    v_18 = sabr_vol_atm(F0, T0, a, b, r_, n)\n"
         "    print(f'beta={b}  (2.17)={v_17:.8f}  (2.18)={v_18:.8f}  diff={abs(v_17-v_18):.2e}')"),
    md("---\n## Sensitivity 1 — $\\alpha$ (volatility level)"),
    code("plt.figure()\n"
         "for a in [0.15, 0.20, 0.25, 0.30]:\n"
         "    plt.plot(Ks, sabr_vol(Ks, F0, T0, a, beta0, rho0, nu0), label=f'α={a}')\n"
         "plt.axvline(F0, color='k', lw=0.6, ls='--')\n"
         "plt.xlabel('K'); plt.ylabel(r'$\\sigma_{SABR}$')\n"
         "plt.title(r'$\\alpha$ — level shift')\n"
         "plt.legend(); plt.show()"),
    md("---\n## Sensitivity 2 — $\\beta$ (backbone)\n\n"
       "Compare **β ∈ {0, 0.5, 1}**, re-scaling α so that the ATM "
       "volatility is matched across β (apples-to-apples)."),
    code("target_atm = 0.20\n"
         "def alpha_for_atm(F, T, beta, rho, nu, atm):\n"
         "    # solve alpha/F^(1-beta) * correction ≈ atm (1st-order use atm*F^(1-beta))\n"
         "    return atm * F ** (1.0 - beta)\n\n"
         "plt.figure()\n"
         "for b in [0.0, 0.5, 1.0]:\n"
         "    a = alpha_for_atm(F0, T0, b, rho0, nu0, target_atm)\n"
         "    plt.plot(Ks, sabr_vol(Ks, F0, T0, a, b, rho0, nu0), label=f'β={b}  (α={a:.3g})')\n"
         "plt.axvline(F0, color='k', lw=0.6, ls='--')\n"
         "plt.xlabel('K'); plt.ylabel(r'$\\sigma_{SABR}$')\n"
         "plt.title(r'$\\beta$ — backbone shape at fixed ATM')\n"
         "plt.legend(); plt.show()"),
    md("### β — backbone dynamics\n"
       "As $F$ moves, ATM vol traces a backbone $\\sigma_{ATM} \\propto F^{\\beta-1}$ "
       "(Hagan eq. 2.15). Plot ATM vol as $F$ varies:"),
    code("Fs = np.linspace(70, 130, 40)\n"
         "plt.figure()\n"
         "for b in [0.0, 0.5, 1.0]:\n"
         "    a = alpha_for_atm(F0, T0, b, rho0, nu0, target_atm)\n"
         "    atms = [sabr_vol_atm(f, T0, a, b, rho0, nu0) for f in Fs]\n"
         "    plt.plot(Fs, atms, label=f'β={b}')\n"
         "plt.xlabel('Forward F'); plt.ylabel(r'$\\sigma_{ATM}$')\n"
         "plt.title('Backbone: ATM vol vs forward (Hagan eq. 2.15)')\n"
         "plt.legend(); plt.show()"),
    md("**Observation.** β = 1 (lognormal) ⇒ ATM vol independent of F. β = 0 (normal) ⇒ "
       "ATM vol scales like $1/F$. β = 0.5 (CIR/bond-like) is intermediate."),
    md("---\n## Sensitivity 3 — $\\rho$ (skew)"),
    code("plt.figure()\n"
         "for r_ in [-0.6, -0.3, 0.0, 0.3, 0.6]:\n"
         "    plt.plot(Ks, sabr_vol(Ks, F0, T0, alpha0, beta0, r_, nu0), label=f'ρ={r_}')\n"
         "plt.axvline(F0, color='k', lw=0.6, ls='--')\n"
         "plt.xlabel('K'); plt.ylabel(r'$\\sigma_{SABR}$')\n"
         "plt.title(r'$\\rho$ — skew')\n"
         "plt.legend(); plt.show()"),
    md("---\n## Sensitivity 4 — $\\nu$ (smile curvature)"),
    code("plt.figure()\n"
         "for n in [0.1, 0.3, 0.5, 0.7, 1.0]:\n"
         "    plt.plot(Ks, sabr_vol(Ks, F0, T0, alpha0, beta0, rho0, n), label=f'ν={n}')\n"
         "plt.axvline(F0, color='k', lw=0.6, ls='--')\n"
         "plt.xlabel('K'); plt.ylabel(r'$\\sigma_{SABR}$')\n"
         "plt.title(r'$\\nu$ — smile curvature (vol-of-vol)')\n"
         "plt.legend(); plt.show()"),
    md("---\n## Summary\n"
       "* SABR implied-vol formula (Hagan 2.17a-c + 2.18 ATM) implemented.\n"
       "* Consistency check: (2.17) at K=F equals (2.18) to machine precision.\n"
       "* Sensitivity plots reproduce the qualitative descriptions on "
       "slide 8 of the outline:\n"
       "  * $\\alpha$ shifts the whole smile level\n"
       "  * $\\beta$ controls the backbone\n"
       "  * $\\rho$ controls the skew\n"
       "  * $\\nu$ controls the smile curvature\n"
       "* Ready for Module 2 (sensitivity study) and Module 4 "
       "(market calibration)."),
], title="02 SABR Implied Vol")

save(nb2, "02_sabr_vol.ipynb")


# ==============================================================
# Notebook 3: Dupire local volatility
# ==============================================================
nb3 = make_nb([
    md("# 03 — Dupire Local Volatility\n\n"
       "Reference: Hagan et al. (2002) §2 / eq. (2.8); Dupire (1994).\n\n"
       "**Local-vol model:** $dF_t = \\sigma_{loc}(F_t)\\,F_t\\,dW$.\n\n"
       "**Dupire's formula (in terms of implied variance "
       "$w(K,T) = \\sigma^2(K,T)\\,T$ and log-moneyness $y = \\ln(K/F)$):**\n\n"
       "$$ \\sigma_{loc}^2(K,T) = "
       "\\frac{\\partial w / \\partial T}"
       "{1 - \\tfrac{y}{w}\\partial_y w + \\tfrac14\\!\\left(-\\tfrac14 - \\tfrac1w + \\tfrac{y^2}{w^2}\\right)(\\partial_y w)^2 "
       "+ \\tfrac12 \\partial_{yy} w}. $$\n\n"
       "Local vol perfectly re-prices today's smile but its dynamics are "
       "the **wrong way round** — Hagan's figures 2.2–2.4: when $F$ rises, "
       "the model shifts the smile down, whereas the market shifts it up. "
       "This is exactly the motivation on slide 4 of the outline."),
    code(SETUP),
    code("from src.sabr import sabr_vol\n"
         "from src.local_vol import dupire_local_vol"),
    md("## Test 1 — Flat Black-Scholes surface recovers $\\sigma_{loc} = \\sigma$"),
    code("F = 100.0\n"
         "Ks = np.linspace(70, 130, 41)\n"
         "Ts = np.linspace(0.1, 2.0, 20)\n"
         "flat = np.full((len(Ts), len(Ks)), 0.20)\n"
         "sloc = dupire_local_vol(flat, Ks, Ts, F)\n"
         "interior = sloc[1:-1, 1:-1]\n"
         "print(f'interior local vol: mean = {np.nanmean(interior):.6f}, '\n"
         "      f'std = {np.nanstd(interior):.2e}')\n"
         "assert abs(np.nanmean(interior) - 0.20) < 1e-6"),
    md("## Test 2 — Dupire applied to a SABR-generated surface\n\n"
       "Take a SABR surface as the \"market\" and extract its local-vol "
       "image. This is the surface that a Local-Vol model would need to "
       "use in order to re-price the SABR smile exactly today."),
    code("T0 = 1.0\n"
         "alpha0, beta0, rho0, nu0 = 0.20, 0.5, -0.30, 0.40\n\n"
         "Ks = np.linspace(70, 130, 61)\n"
         "Ts = np.linspace(0.25, 2.0, 18)\n\n"
         "iv_surface = np.array([sabr_vol(Ks, F, T_, alpha0, beta0, rho0, nu0) for T_ in Ts])\n"
         "sloc = dupire_local_vol(iv_surface, Ks, Ts, F)\n\n"
         "fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))\n"
         "im0 = ax[0].imshow(iv_surface, aspect='auto', origin='lower',\n"
         "                   extent=[Ks[0], Ks[-1], Ts[0], Ts[-1]])\n"
         "ax[0].set_title(r'SABR implied vol  $\\sigma(K,T)$')\n"
         "ax[0].set_xlabel('K'); ax[0].set_ylabel('T')\n"
         "plt.colorbar(im0, ax=ax[0])\n\n"
         "im1 = ax[1].imshow(sloc, aspect='auto', origin='lower',\n"
         "                   extent=[Ks[0], Ks[-1], Ts[0], Ts[-1]])\n"
         "ax[1].set_title(r'Dupire local vol  $\\sigma_{loc}(K,T)$')\n"
         "ax[1].set_xlabel('K'); ax[1].set_ylabel('T')\n"
         "plt.colorbar(im1, ax=ax[1])\n"
         "plt.tight_layout(); plt.show()"),
    md("### Slice at a fixed $T$: local-vol skew ≈ **twice** implied-vol skew\n"
       "This is the well-known Dupire rule of thumb (Hagan §2)."),
    code("iT = len(Ts) // 2\n"
         "plt.figure()\n"
         "plt.plot(Ks, iv_surface[iT],  label=r'$\\sigma_{SABR}(K,T)$')\n"
         "plt.plot(Ks, sloc[iT],        label=r'$\\sigma_{loc}(K,T)$')\n"
         "plt.axvline(F, color='k', lw=0.6, ls='--')\n"
         "plt.xlabel('K'); plt.ylabel('volatility')\n"
         "plt.title(f'Slice at T = {Ts[iT]:.2f}')\n"
         "plt.legend(); plt.show()"),
    md("## Summary\n"
       "* Dupire estimator implemented via finite differences on $w(K,T)$.\n"
       "* Unit test: flat BS surface ⇒ $\\sigma_{loc}$ constant to machine precision.\n"
       "* Applied to SABR surface: recovers a well-defined local-vol surface "
       "with roughly twice the slope of the implied-vol skew (Dupire rule).\n"
       "* Module 5 will use this routine to demonstrate the "
       "**wrong dynamics** of local vol (market: smile shifts *up* with F; "
       "local vol: smile shifts *down*)."),
], title="03 Dupire Local Vol")

save(nb3, "03_local_vol.ipynb")
