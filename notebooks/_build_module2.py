"""Builder for the Module-2 notebook: SABR parameter sensitivity study.

Run from the `notebooks/` directory:
    python3 _build_module2.py
"""
import json

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


SETUP = r"""import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.figsize': (7, 4.5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})

from src.sabr import sabr_vol, sabr_vol_atm
from src.sensitivity import (
    atm_level, atm_slope, atm_curvature,
    backbone, alpha_for_target_atm,
)
"""


cells = [
    md(
        "# 04 — Parameter Sensitivity of the SABR Smile\n\n"
        "**Goal.** For each of the four SABR parameters "
        "$(\\alpha, \\beta, \\rho, \\nu)$, show *how* it deforms the "
        "implied-volatility smile and quantify the effect. "
        "This delivers the *Parameter Sensitivity* analysis described on "
        "slides 8–9 of the outline.\n\n"
        "**Plan.**\n"
        "1. Baseline smile $\\sigma(K)$ (reproduces Hagan fig. 3.3 shape).\n"
        "2. $\\alpha$ — volatility **level**.\n"
        "3. $\\beta$ — **backbone** (static shape + $\\sigma_{ATM}$ vs $F$).\n"
        "4. $\\rho$ — **skew**.\n"
        "5. $\\nu$ — **smile curvature**.\n"
        "6. Joint $(\\rho, \\nu)$ heatmap.\n"
        "7. Term-structure: smile flattening with $T$.\n"
        "8. Quantitative summary table.\n\n"
        "The smile is summarized by three ATM descriptors\n"
        "$$\\text{level} = \\sigma(F,F,T),\\quad "
        "\\text{slope} = \\partial_{\\ln K}\\sigma|_{K=F},\\quad "
        "\\text{curv.} = \\partial^2_{\\ln K}\\sigma|_{K=F}.$$"
    ),
    code(SETUP),
    code(
        "# --- Baseline market state (equity-index-like) ---\n"
        "F0 = 100.0        # forward\n"
        "T0 = 1.0          # time to expiry (years)\n"
        "alpha0 = 0.20     # SABR alpha\n"
        "beta0  = 0.5      # elasticity\n"
        "rho0   = -0.30    # correlation\n"
        "nu0    = 0.40     # vol-of-vol\n\n"
        "Ks = np.linspace(70, 130, 121)\n"
        "base_iv = sabr_vol(Ks, F0, T0, alpha0, beta0, rho0, nu0)\n\n"
        "plt.plot(Ks, base_iv, lw=2)\n"
        "plt.axvline(F0, color='k', lw=0.8, ls='--', label='F')\n"
        "plt.xlabel('Strike K'); plt.ylabel(r'$\\sigma_{SABR}$')\n"
        "plt.title(rf'Baseline smile  $\\alpha$={alpha0}, $\\beta$={beta0}, '\n"
        "          rf'$\\rho$={rho0}, $\\nu$={nu0},  T={T0}')\n"
        "plt.legend(); plt.show()\n\n"
        "print(f'ATM level     = {atm_level(F0,T0,alpha0,beta0,rho0,nu0):.5f}')\n"
        "print(f'ATM slope     = {atm_slope(F0,T0,alpha0,beta0,rho0,nu0):.5f}')\n"
        "print(f'ATM curvature = {atm_curvature(F0,T0,alpha0,beta0,rho0,nu0):.5f}')"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 1. $\\alpha$ — volatility level\n\n"
        "$\\alpha$ sets the overall level of the smile; it enters as a "
        "multiplicative factor in Hagan (2.17a). Expect an approximately "
        "rigid vertical translation."
    ),
    code(
        "alpha_grid = [0.10, 0.15, 0.20, 0.25, 0.30]\n\n"
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))\n\n"
        "for a in alpha_grid:\n"
        "    ax[0].plot(Ks, sabr_vol(Ks, F0, T0, a, beta0, rho0, nu0), label=rf'$\\alpha$={a}')\n"
        "ax[0].axvline(F0, color='k', lw=0.6, ls='--')\n"
        "ax[0].set_xlabel('K'); ax[0].set_ylabel(r'$\\sigma$')\n"
        "ax[0].set_title(r'Smiles across $\\alpha$'); ax[0].legend()\n\n"
        "levels = [atm_level(F0,T0,a,beta0,rho0,nu0) for a in alpha_grid]\n"
        "ax[1].plot(alpha_grid, levels, 'o-')\n"
        "ax[1].set_xlabel(r'$\\alpha$'); ax[1].set_ylabel(r'$\\sigma_{ATM}$')\n"
        "ax[1].set_title(r'ATM level vs $\\alpha$ (near-linear)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 2. $\\beta$ — backbone\n\n"
        "$\\beta$ has two distinct effects:\n"
        "* **Shape** of the smile at a fixed forward: different $\\beta$ "
        "  produce different curvature/wing behaviour.\n"
        "* **Backbone**: as $F$ moves, the ATM vol traces "
        "  $\\sigma_{ATM} \\sim \\alpha / F^{\\,1-\\beta}$ (Hagan 2.15).\n"
        "  $\\beta = 1$ (lognormal) ⇒ flat backbone — equity/FX style.\n"
        "  $\\beta = 0$ (normal)  ⇒ $\\sigma_{ATM} \\propto 1/F$ — rate style.\n\n"
        "We compare **$\\beta \\in \\{0, 0.5, 1\\}$** with $\\alpha$ re-scaled "
        "so that the ATM volatility is identical across $\\beta$ "
        "(apples-to-apples comparison)."
    ),
    code(
        "beta_grid = [0.0, 0.5, 1.0]\n"
        "target_atm = 0.20\n\n"
        "alphas = {b: alpha_for_target_atm(F0, T0, b, rho0, nu0, target_atm) for b in beta_grid}\n"
        "for b, a in alphas.items():\n"
        "    atm = atm_level(F0, T0, a, b, rho0, nu0)\n"
        "    print(f'beta={b}:  alpha={a:.4g},  ATM={atm:.6f}')"
    ),
    md("### 2a. Static smile shape across $\\beta$"),
    code(
        "plt.figure()\n"
        "for b in beta_grid:\n"
        "    plt.plot(Ks, sabr_vol(Ks, F0, T0, alphas[b], b, rho0, nu0),\n"
        "             label=rf'$\\beta$={b}  ($\\alpha$={alphas[b]:.3g})', lw=1.8)\n"
        "plt.axvline(F0, color='k', lw=0.6, ls='--')\n"
        "plt.xlabel('K'); plt.ylabel(r'$\\sigma$')\n"
        "plt.title(r'Static smile shape across $\\beta$ (matched ATM)')\n"
        "plt.legend(); plt.show()"
    ),
    md(
        "### 2b. Backbone — $\\sigma_{ATM}$ as $F$ moves\n"
        "For each $\\beta$, freeze $(\\alpha, \\rho, \\nu)$ at the matched-ATM "
        "values above and sweep $F$. The resulting locus is the **backbone**."
    ),
    code(
        "F_grid = np.linspace(70, 130, 61)\n"
        "plt.figure()\n"
        "for b in beta_grid:\n"
        "    bb = backbone(F_grid, T0, alphas[b], b, rho0, nu0)\n"
        "    plt.plot(F_grid, bb, label=rf'$\\beta$={b}', lw=1.8)\n"
        "plt.axvline(F0, color='k', lw=0.6, ls='--')\n"
        "plt.axhline(target_atm, color='grey', lw=0.6, ls=':')\n"
        "plt.xlabel('Forward F'); plt.ylabel(r'$\\sigma_{ATM}$')\n"
        "plt.title(r'Backbone: ATM vol vs F for different $\\beta$')\n"
        "plt.legend(); plt.show()"
    ),
    md(
        "**Reading the plot.**\n"
        "* $\\beta = 1$: essentially flat — classical Black/lognormal.\n"
        "* $\\beta = 0$: steep negative slope — normal model; vol inversely scales with F.\n"
        "* $\\beta = 0.5$: intermediate; the shape used in the paper for interest-rate smiles.\n\n"
        "On slide 4 of the outline the point is that **local volatility** locks the "
        "backbone into a deterministic function of F, giving the *wrong* direction "
        "empirically; SABR, by contrast, is a stochastic-vol model where the "
        "backbone shape is controlled by $\\beta$ and the correlation $\\rho$ "
        "separately."
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 3. $\\rho$ — skew\n\n"
        "$\\rho$ controls the correlation between the forward and its "
        "stochastic vol. Empirically, equity indices have $\\rho < 0$ "
        "(vol rises when the market falls), producing a **negative ATM slope**."
    ),
    code(
        "rho_grid = [-0.7, -0.4, -0.1, 0.1, 0.4, 0.7]\n\n"
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))\n"
        "for r in rho_grid:\n"
        "    ax[0].plot(Ks, sabr_vol(Ks, F0, T0, alpha0, beta0, r, nu0), label=rf'$\\rho$={r}')\n"
        "ax[0].axvline(F0, color='k', lw=0.6, ls='--')\n"
        "ax[0].set_xlabel('K'); ax[0].set_ylabel(r'$\\sigma$')\n"
        "ax[0].set_title(r'Smile vs $\\rho$'); ax[0].legend(fontsize=9)\n\n"
        "slopes = [atm_slope(F0,T0,alpha0,beta0,r,nu0) for r in rho_grid]\n"
        "ax[1].plot(rho_grid, slopes, 'o-', color='C3')\n"
        "ax[1].axhline(0, color='k', lw=0.6)\n"
        "ax[1].set_xlabel(r'$\\rho$'); ax[1].set_ylabel(r'ATM slope $\\partial_{\\ln K}\\sigma$')\n"
        "ax[1].set_title(r'ATM slope is monotone in $\\rho$')\n"
        "plt.tight_layout(); plt.show()"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 4. $\\nu$ — smile curvature (vol-of-vol)\n\n"
        "$\\nu$ controls how variable the stochastic vol is. "
        "Larger $\\nu$ ⇒ fatter wings ⇒ higher ATM **curvature**."
    ),
    code(
        "nu_grid = [0.05, 0.20, 0.40, 0.60, 0.80, 1.00]\n\n"
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))\n"
        "for n in nu_grid:\n"
        "    ax[0].plot(Ks, sabr_vol(Ks, F0, T0, alpha0, beta0, rho0, n), label=rf'$\\nu$={n}')\n"
        "ax[0].axvline(F0, color='k', lw=0.6, ls='--')\n"
        "ax[0].set_xlabel('K'); ax[0].set_ylabel(r'$\\sigma$')\n"
        "ax[0].set_title(r'Smile vs $\\nu$'); ax[0].legend(fontsize=9)\n\n"
        "curvs = [atm_curvature(F0,T0,alpha0,beta0,rho0,n) for n in nu_grid]\n"
        "ax[1].plot(nu_grid, curvs, 'o-', color='C2')\n"
        "ax[1].set_xlabel(r'$\\nu$'); ax[1].set_ylabel(r'ATM curvature $\\partial^2_{\\ln K}\\sigma$')\n"
        "ax[1].set_title(r'Smile curvature grows with $\\nu$')\n"
        "plt.tight_layout(); plt.show()"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 5. Joint $(\\rho, \\nu)$ effect\n\n"
        "Two heatmaps over a $(\\rho, \\nu)$ grid: ATM **slope** (skew) "
        "and ATM **curvature** (smile). Confirms that $\\rho$ dominates "
        "the slope and $\\nu$ dominates the curvature, with mild cross-terms."
    ),
    code(
        "rho_ax = np.linspace(-0.8, 0.8, 33)\n"
        "nu_ax  = np.linspace(0.05, 1.0, 20)\n"
        "RR, NN = np.meshgrid(rho_ax, nu_ax)\n"
        "SLOPE = np.vectorize(lambda r, n: atm_slope(F0,T0,alpha0,beta0,r,n))(RR, NN)\n"
        "CURV  = np.vectorize(lambda r, n: atm_curvature(F0,T0,alpha0,beta0,r,n))(RR, NN)\n\n"
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "im0 = ax[0].pcolormesh(rho_ax, nu_ax, SLOPE, shading='auto', cmap='RdBu_r',\n"
        "                       vmin=-abs(SLOPE).max(), vmax=abs(SLOPE).max())\n"
        "ax[0].set_xlabel(r'$\\rho$'); ax[0].set_ylabel(r'$\\nu$')\n"
        "ax[0].set_title('ATM slope (skew)'); plt.colorbar(im0, ax=ax[0])\n\n"
        "im1 = ax[1].pcolormesh(rho_ax, nu_ax, CURV, shading='auto', cmap='viridis')\n"
        "ax[1].set_xlabel(r'$\\rho$'); ax[1].set_ylabel(r'$\\nu$')\n"
        "ax[1].set_title('ATM curvature (smile)'); plt.colorbar(im1, ax=ax[1])\n"
        "plt.tight_layout(); plt.show()"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 6. Term structure\n\n"
        "With all SABR parameters held fixed, sweep the expiry $T$. "
        "SABR's leading-order correction in Hagan (2.17) is $\\mathcal{O}(T)$, "
        "so the smile flattens as $T$ grows."
    ),
    code(
        "T_grid = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0]\n"
        "plt.figure()\n"
        "for T_ in T_grid:\n"
        "    plt.plot(Ks, sabr_vol(Ks, F0, T_, alpha0, beta0, rho0, nu0), label=f'T={T_}')\n"
        "plt.axvline(F0, color='k', lw=0.6, ls='--')\n"
        "plt.xlabel('K'); plt.ylabel(r'$\\sigma$')\n"
        "plt.title('Smile term structure')\n"
        "plt.legend(); plt.show()\n\n"
        "# Quantitative: slope and curvature as functions of T\n"
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))\n"
        "Ts = np.linspace(0.05, 3.0, 60)\n"
        "sls = [atm_slope(F0,T,alpha0,beta0,rho0,nu0) for T in Ts]\n"
        "cvs = [atm_curvature(F0,T,alpha0,beta0,rho0,nu0) for T in Ts]\n"
        "ax[0].plot(Ts, sls); ax[0].set_xlabel('T'); ax[0].set_ylabel('ATM slope')\n"
        "ax[0].set_title('Skew decays with T')\n"
        "ax[1].plot(Ts, cvs, color='C2'); ax[1].set_xlabel('T'); ax[1].set_ylabel('ATM curvature')\n"
        "ax[1].set_title('Curvature decays with T')\n"
        "plt.tight_layout(); plt.show()"
    ),
    # -----------------------------------------------------------
    md("---\n## 7. Quantitative summary"),
    code(
        "# Sweep one parameter at a time around the baseline; report descriptors.\n"
        "rows = []\n"
        "def describe(name, val, a, b, r, n):\n"
        "    rows.append({\n"
        "        'varied': name,\n"
        "        'value' : val,\n"
        "        'ATM level'    : atm_level(F0, T0, a, b, r, n),\n"
        "        'ATM slope'    : atm_slope(F0, T0, a, b, r, n),\n"
        "        'ATM curvature': atm_curvature(F0, T0, a, b, r, n),\n"
        "    })\n\n"
        "for a in alpha_grid:  describe('alpha', a, a, beta0, rho0, nu0)\n"
        "for b in beta_grid:   describe('beta' , b, alphas[b], b, rho0, nu0)\n"
        "for r in rho_grid:    describe('rho'  , r, alpha0, beta0, r, nu0)\n"
        "for n in nu_grid:     describe('nu'   , n, alpha0, beta0, rho0, n)\n\n"
        "df = pd.DataFrame(rows)\n"
        "df"
    ),
    md(
        "---\n## Summary\n\n"
        "| Parameter | Dominant effect | Observed signature |\n"
        "|---|---|---|\n"
        "| $\\alpha$ | level | ATM level $\\uparrow$ roughly linearly in $\\alpha$ |\n"
        "| $\\beta$  | backbone | Flat ($\\beta{=}1$) / decreasing ($\\beta{<}1$) ATM vs $F$ |\n"
        "| $\\rho$   | skew  | ATM slope is monotone in $\\rho$; $\\rho{<}0$ ⇒ negative skew |\n"
        "| $\\nu$    | smile | ATM curvature increases with $\\nu$ |\n\n"
        "Matches slide 8 in the outline qualitatively and "
        "gives the numerical backing for the parameter-intuition story told in "
        "the presentation. These descriptors will also be the **calibration "
        "targets** in Module 4: "
        "$\\rho$ is pinned by the observed ATM skew, $\\nu$ by the observed "
        "curvature, $\\alpha$ by the ATM level; $\\beta$ is usually fixed by "
        "market convention (0.5 for rates, often ≈1 for equity indices)."
    ),
]

nb = make_nb(cells, "04 Parameter Sensitivity")
save(nb, "04_parameter_sensitivity.ipynb")
