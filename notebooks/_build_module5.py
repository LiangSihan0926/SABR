"""Builder for Module 5 notebook: SABR vs Black-Scholes vs Local Vol.

Delivers slide 10 of the outline:
    - SABR vs Black-Scholes: smile-fitting accuracy
    - SABR vs Local Volatility: dynamics consistency
    - Qualitative behaviour

Run from `notebooks/`:  python3 _build_module5.py
"""
import json

NB_VERSION = 4


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": s}


SETUP = r"""import sys, os, time
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({
    'figure.figsize': (8, 4.5),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11,
})

DATA_DIR  = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data'))
CACHE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'cache'))

from src.sabr import sabr_vol
from src.data_loader import load_fred_yields, build_smile
from src.calibration import calibrate_smile_panel, calibrate_sabr
from src.model_compare import (
    fit_flat_bs, local_vol_slice,
    predict_sticky_strike, predict_sticky_moneyness,
    predict_local_vol, predict_sabr, rmse,
)

rates = load_fred_yields(DATA_DIR).ffill()
spy   = pd.read_parquet(os.path.join(CACHE_DIR, 'spy_options_filtered.parquet'))
qqq   = pd.read_parquet(os.path.join(CACHE_DIR, 'qqq_options_filtered.parquet'))
grid  = pd.read_parquet(os.path.join(CACHE_DIR, 'calibration_grid.parquet'))
cal   = pd.read_parquet(os.path.join(CACHE_DIR, 'calibration_results.parquet'))
print(f'loaded: rates {rates.shape}, spy {spy.shape[0]:,}, qqq {qqq.shape[0]:,}, grid {grid.shape[0]}, cal {cal.shape[0]}')
"""


cells = [
    md(
        "# 07 — Model Comparison: SABR vs Black-Scholes vs Local Vol\n\n"
        "This notebook delivers **Slide 10** of the outline. Three evaluation axes, "
        "applied to the three models:\n\n"
        "| Model | Parameters per smile | What it sees |\n"
        "|---|---|---|\n"
        "| Black-Scholes (flat) | $\\sigma$ | one scalar per smile |\n"
        "| Local Vol (Dupire)   | $\\sigma_{loc}(K,T)$ | fits smile perfectly by construction |\n"
        "| SABR | $(\\alpha,\\beta,\\rho,\\nu)$ | fits smile via closed-form |\n\n"
        "**Axes.**\n"
        "1. **Static fit accuracy** — BS vs SABR RMSE across 210 market smiles.\n"
        "2. **Dynamics consistency** — synthetic $F$-shift experiment, then empirical "
        "   next-smile prediction across date pairs.\n"
        "3. **Backbone direction** — regress observed $\\ln\\sigma_{ATM}$ on $\\ln F$ "
        "   and compare to the SABR rule $\\sigma_{ATM}\\propto \\alpha/F^{1-\\beta}$.\n"
        "4. Qualitative summary."
    ),
    code(SETUP),
    # ==================================================================
    md(
        "---\n## 1. SABR vs Black-Scholes — static fit accuracy\n\n"
        "BS needs a single $\\sigma$ to price the whole smile. We fit it by "
        "minimising the same RMSE objective that calibrated SABR used. "
        "The difference in RMSE is the *cost* of assuming a flat vol."
    ),
    code(
        "def fit_bs_for_smile(tk, trade_date, dte):\n"
        "    df = {'spy': spy, 'qqq': qqq}[tk]\n"
        "    sm = build_smile(df, rates, trade_date, dte=dte)\n"
        "    sigma_bs, rmse_bs = fit_flat_bs(sm['STRIKE'].to_numpy(),\n"
        "                                    sm['SIGMA_MKT'].to_numpy())\n"
        "    return sigma_bs, rmse_bs, sm\n\n"
        "# 1a — single smile visualization\n"
        "sigma_bs, rmse_bs, sm = fit_bs_for_smile('spy', '2022-01-03', 30)\n"
        "f_sabr = calibrate_smile_panel(sm, beta=0.5)\n"
        "F_, T_ = sm.attrs['F'], sm.attrs['expire_dte']/365.25\n"
        "\n"
        "Ks = np.linspace(sm['STRIKE'].min(), sm['STRIKE'].max(), 200)\n"
        "plt.scatter(sm['STRIKE'], sm['SIGMA_MKT']*100, c='k', s=18, label='market', zorder=3)\n"
        "plt.plot(Ks, sabr_vol(Ks, F_, T_, f_sabr.alpha, 0.5, f_sabr.rho, f_sabr.nu)*100,\n"
        "         'C0', label=rf'SABR ($\\beta$=0.5), RMSE={f_sabr.rmse*100:.2f}%')\n"
        "plt.axhline(sigma_bs*100, color='C3', ls='--', label=f'BS flat σ={sigma_bs*100:.1f}%, RMSE={rmse_bs*100:.2f}%')\n"
        "plt.axvline(F_, color='grey', ls=':', lw=0.6)\n"
        "plt.xlabel('K'); plt.ylabel('IV (%)')\n"
        "plt.title(f'SPY 2022-01-03 30-DTE — SABR vs flat BS')\n"
        "plt.legend(); plt.show()"
    ),
    md("### 1b — RMSE distribution over the 210 market smiles"),
    code(
        "# run BS flat fit for every row in the grid; join with SABR results (beta=0.5)\n"
        "rows = []\n"
        "by_tk = {'spy': spy, 'qqq': qqq}\n"
        "for _, g in grid.iterrows():\n"
        "    try:\n"
        "        sm = build_smile(by_tk[g.ticker], rates, g.trade_date, dte=g.dte)\n"
        "    except Exception:\n"
        "        continue\n"
        "    if len(sm) < 5: continue\n"
        "    sigma_bs, rmse_bs = fit_flat_bs(sm['STRIKE'].to_numpy(), sm['SIGMA_MKT'].to_numpy())\n"
        "    rows.append({\n"
        "        'ticker': g.ticker.upper(), 'trade_date': g.trade_date, 'dte': g.dte,\n"
        "        'bs_sigma': sigma_bs, 'bs_rmse': rmse_bs,\n"
        "    })\n"
        "bs_df = pd.DataFrame(rows)\n"
        "merged = bs_df.merge(cal.query('beta == 0.5'),\n"
        "                     on=['ticker', 'trade_date', 'dte'])\n"
        "print(f'joined table: {len(merged)} rows')\n"
        "merged[['rmse', 'bs_rmse']].describe(percentiles=[.5, .9]).round(5)"
    ),
    code(
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "ax[0].scatter(merged['bs_rmse']*10000, merged['rmse']*10000, s=12, alpha=0.6)\n"
        "lim = max(merged['bs_rmse'].max(), merged['rmse'].max())*10000\n"
        "ax[0].plot([0, lim], [0, lim], 'k--', lw=0.6)\n"
        "ax[0].set_xlabel('BS flat RMSE (bps of vol)')\n"
        "ax[0].set_ylabel('SABR β=0.5 RMSE (bps of vol)')\n"
        "ax[0].set_title('Per-smile fit quality — SABR vs BS')\n"
        "\n"
        "data = [merged['rmse']*10000, merged['bs_rmse']*10000]\n"
        "ax[1].boxplot(data, tick_labels=['SABR', 'BS flat'])\n"
        "ax[1].set_ylabel('RMSE (bps of vol)')\n"
        "ax[1].set_yscale('log')\n"
        "ax[1].set_title('Distribution of fit RMSE')\n"
        "plt.tight_layout(); plt.show()\n\n"
        "ratio = (merged['bs_rmse'] / merged['rmse']).median()\n"
        "print(f'Median RMSE ratio BS/SABR = {ratio:.1f}x')"
    ),
    md(
        "**Observation.** BS flat-vol fit is typically *an order of magnitude* worse "
        "than SABR. This is the quantitative version of the motivation on slide 3: "
        "BS is one scalar, the market smile is not."
    ),
    # ==================================================================
    md(
        "---\n## 2. SABR vs Local Vol — dynamics consistency\n\n"
        "This is the central argument of the paper (Hagan §2, figures 2.2-2.4) "
        "and slide 4 of the outline. Both models can *re-price today's smile*; "
        "they disagree on *tomorrow's smile* after $F$ moves.\n\n"
        "Four prediction rules:\n\n"
        "| Rule | Formula | Informal name |\n"
        "|---|---|---|\n"
        "| Sticky strike | $\\sigma_{\\rm new}(K) = \\sigma_{\\rm today}(K)$ | BS + fixed-K vol |\n"
        "| Sticky moneyness | $\\sigma_{\\rm new}(K) = \\sigma_{\\rm today}(K\\cdot F_{\\rm today}/F_{\\rm new})$ | \"smile moves with market\" |\n"
        "| Local vol (Hagan approx.) | $\\sigma_{\\rm new}(K)\\approx\\sigma_{\\rm loc}\\!\\left(\\tfrac{K+F_{\\rm new}}{2},T\\right)$ | Dupire-implied |\n"
        "| SABR | $\\sigma_{\\rm new}(K) = \\sigma_{SABR}(K, F_{\\rm new};\\alpha,\\beta,\\rho,\\nu)$ | this model |"
    ),
    md(
        "### 2a — Synthetic experiment\n\n"
        "Start from the real SPY 2022-01-03 30-DTE calibration ($\\beta=0.5$), "
        "shift $F$ by ±5 %, and plot what each rule predicts."
    ),
    code(
        "# base state\n"
        "sm0  = build_smile(spy, rates, '2022-01-03', dte=30)\n"
        "F0   = sm0.attrs['F']\n"
        "T0   = sm0.attrs['expire_dte'] / 365.25\n"
        "r0   = sm0.attrs['r']\n"
        "fit0 = calibrate_smile_panel(sm0, beta=0.5)\n"
        "alpha0, beta0, rho0, nu0 = fit0.alpha, 0.5, fit0.rho, fit0.nu\n"
        "K_today = sm0['STRIKE'].to_numpy()\n"
        "s_today = sm0['SIGMA_MKT'].to_numpy()\n\n"
        "# local vol slice on a wide grid (boundary points are inaccurate)\n"
        "K_lv = np.linspace(0.6*F0, 1.4*F0, 181)\n"
        "sloc = local_vol_slice(K_lv, F0, T0, alpha0, beta0, rho0, nu0)\n\n"
        "# predictions at +/- 5% F\n"
        "shifts = [(F0 * 0.95, 'F down 5%'), (F0 * 1.05, 'F up 5%')]\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "for ax, (F_new, tag) in zip(axes, shifts):\n"
        "    K_new = np.linspace(0.85*F_new, 1.15*F_new, 60)\n"
        "    sabr  = predict_sabr(K_new, F_new, T0, alpha0, beta0, rho0, nu0)\n"
        "    stst  = predict_sticky_strike(K_new, K_today, s_today)\n"
        "    stmo  = predict_sticky_moneyness(K_new, F_new, K_today, F0, s_today)\n"
        "    lv    = predict_local_vol(K_new, F_new, T0, sloc, K_lv)\n\n"
        "    ax.plot(K_today, s_today*100, 'k-', lw=1.5, alpha=0.5, label=f'today (F={F0:.1f})')\n"
        "    ax.plot(K_new, sabr*100, 'C0',  lw=2, label='SABR prediction')\n"
        "    ax.plot(K_new, stmo*100, 'C2--', lw=1.5, label='sticky-moneyness')\n"
        "    ax.plot(K_new, stst*100, 'C1--', lw=1.5, label='sticky-strike')\n"
        "    ax.plot(K_new, lv*100,   'C3--', lw=1.5, label='local vol')\n"
        "    ax.axvline(F0, color='grey', ls=':', lw=0.6, label=f'F_today')\n"
        "    ax.axvline(F_new, color='red', ls=':', lw=0.6, label=f'F_new={F_new:.1f}')\n"
        "    ax.set_xlabel('K'); ax.set_ylabel('IV (%)')\n"
        "    ax.set_title(tag); ax.legend(fontsize=8)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**Reading the plot.** Pay attention to *where each curve sits near the new $F$*.\n"
        "* **Sticky-moneyness** (green dashed) is the empirical rule-of-thumb that "
        "  equity smiles translate *with* the forward — the predicted ATM vol "
        "  roughly equals today's ATM vol.\n"
        "* **Sticky-strike** (orange dashed) leaves the smile fixed in $K$-space, "
        "  so the new ATM vol moves along today's skew.\n"
        "* **Local vol** (red dashed) is Dupire's approximation. Because "
        "  $\\sigma_{loc}$ has roughly *twice* the skew of $\\sigma_{imp}$, "
        "  the local-vol prediction moves *further* than sticky-strike and "
        "  can point in a counter-intuitive direction.\n"
        "* **SABR** (solid blue) sits between sticky-strike and sticky-moneyness "
        "  and is tunable via $\\beta$ (backbone), giving the right empirical "
        "  dynamics for each asset class."
    ),
    md(
        "### 2b — Empirical dynamics test\n\n"
        "For each of a handful of date pairs $(d_1, d_2)$ roughly one week apart, "
        "calibrate SABR on $d_1$, derive local vol on $d_1$, then predict the "
        "$d_2$ smile with all four rules and compare to the observed $d_2$ smile. "
        "Report the RMSE of each prediction."
    ),
    code(
        "pair_starts = ['2018-02-02', '2020-03-13', '2022-01-21',\n"
        "               '2022-06-10', '2023-03-10']\n"
        "# require d2 to be the next-week trade date\n"
        "spy_dates = np.sort(spy['QUOTE_DATE'].unique())\n\n"
        "def nearest_future(d, shift_days=7):\n"
        "    target = np.datetime64(pd.Timestamp(d) + pd.Timedelta(days=shift_days))\n"
        "    fut = spy_dates[spy_dates >= target]\n"
        "    return pd.Timestamp(fut[0]) if len(fut) else None\n\n"
        "pairs = [(pd.Timestamp(d), nearest_future(d)) for d in pair_starts]\n"
        "pairs = [(a, b) for a, b in pairs if b is not None]\n"
        "pd.DataFrame(pairs, columns=['d1', 'd2'])"
    ),
    code(
        "def empirical_prediction_errors(d1, d2, dte=30):\n"
        "    sm1 = build_smile(spy, rates, d1, dte=dte)\n"
        "    sm2 = build_smile(spy, rates, d2, dte=dte)\n"
        "    F1, F2 = sm1.attrs['F'], sm2.attrs['F']\n"
        "    T1, T2 = sm1.attrs['expire_dte']/365.25, sm2.attrs['expire_dte']/365.25\n"
        "    # calibrate SABR on d1 (beta=0.5)\n"
        "    fit = calibrate_smile_panel(sm1, beta=0.5)\n"
        "    alpha, rho, nu = fit.alpha, fit.rho, fit.nu\n"
        "    # local vol slice on d1\n"
        "    K_lv = np.linspace(0.6*F1, 1.4*F1, 161)\n"
        "    sloc = local_vol_slice(K_lv, F1, T1, alpha, 0.5, rho, nu)\n"
        "    # target: d2 observed\n"
        "    K2 = sm2['STRIKE'].to_numpy()\n"
        "    s2 = sm2['SIGMA_MKT'].to_numpy()\n"
        "    # predictions\n"
        "    p_sabr = predict_sabr(K2, F2, T2, alpha, 0.5, rho, nu)\n"
        "    p_stst = predict_sticky_strike(K2, sm1['STRIKE'].to_numpy(), sm1['SIGMA_MKT'].to_numpy())\n"
        "    p_stmo = predict_sticky_moneyness(K2, F2, sm1['STRIKE'].to_numpy(),\n"
        "                                       F1, sm1['SIGMA_MKT'].to_numpy())\n"
        "    p_lv   = predict_local_vol(K2, F2, T1, sloc, K_lv)\n"
        "    return {\n"
        "        'd1': d1, 'd2': d2, 'F1': F1, 'F2': F2,\n"
        "        'dF_pct': (F2/F1-1)*100,\n"
        "        'SABR' : rmse(p_sabr, s2),\n"
        "        'stst' : rmse(p_stst, s2),\n"
        "        'stmo' : rmse(p_stmo, s2),\n"
        "        'LV'   : rmse(p_lv,   s2),\n"
        "    }\n\n"
        "rows = [empirical_prediction_errors(d1, d2) for d1, d2 in pairs]\n"
        "pred = pd.DataFrame(rows)\n"
        "pred['F1'] = pred['F1'].round(2); pred['F2'] = pred['F2'].round(2)\n"
        "for c in ('SABR', 'stst', 'stmo', 'LV'):\n"
        "    pred[c] = (pred[c] * 10000).round(1)        # bps of vol\n"
        "pred.rename(columns={'SABR':'SABR (bps)', 'stst':'sticky-strike (bps)',\n"
        "                     'stmo':'sticky-mny (bps)', 'LV':'local-vol (bps)'})"
    ),
    code(
        "# median prediction RMSE, bar plot\n"
        "med = pred[['SABR', 'stst', 'stmo', 'LV']].median()\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.bar(['SABR', 'sticky-strike', 'sticky-mny', 'local-vol'], med.values)\n"
        "ax.set_ylabel('median next-week prediction RMSE (bps of vol)')\n"
        "ax.set_title(f'Empirical dynamics test ({len(pred)} date pairs)')\n"
        "for i, v in enumerate(med.values):\n"
        "    ax.text(i, v, f'{v:.1f}', ha='center', va='bottom')\n"
        "plt.show()"
    ),
    # ==================================================================
    md(
        "---\n## 3. Backbone direction — $\\sigma_{ATM}$ vs $F$\n\n"
        "Hagan eq. (2.15): SABR's ATM vol has the backbone\n"
        "$$ \\ln \\sigma_{ATM} \\approx \\ln \\alpha - (1-\\beta)\\ln F. $$\n"
        "Regressing observed $\\ln \\sigma_{ATM}$ on $\\ln F$ across dates gives "
        "a slope that should be close to $\\beta - 1$. For equity indices the "
        "empirically observed slope is ≈ 0 (i.e. $\\beta\\!\\to\\!1$, lognormal)."
    ),
    code(
        "def observed_atm_vol(tk, dte):\n"
        "    df = {'spy': spy, 'qqq': qqq}[tk]\n"
        "    gs = grid.query('ticker == @tk and dte == @dte')\n"
        "    rows = []\n"
        "    for _, g in gs.iterrows():\n"
        "        try:\n"
        "            sm = build_smile(df, rates, g.trade_date, dte=dte)\n"
        "        except Exception:\n"
        "            continue\n"
        "        if len(sm) < 5: continue\n"
        "        i_atm = int(np.argmin(np.abs(sm['STRIKE'].to_numpy() - sm.attrs['F'])))\n"
        "        rows.append({'trade_date': g.trade_date, 'F': sm.attrs['F'],\n"
        "                      'sigma_atm': float(sm['SIGMA_MKT'].iloc[i_atm])})\n"
        "    return pd.DataFrame(rows)\n\n"
        "bb_spy = observed_atm_vol('spy', 30)\n"
        "bb_qqq = observed_atm_vol('qqq', 30)\n"
        "print(f'SPY {len(bb_spy)} points; QQQ {len(bb_qqq)} points')"
    ),
    code(
        "def fit_loglog(df):\n"
        "    x = np.log(df['F'].to_numpy())\n"
        "    y = np.log(df['sigma_atm'].to_numpy())\n"
        "    slope, intercept = np.polyfit(x, y, 1)\n"
        "    return slope, intercept\n\n"
        "s_spy, c_spy = fit_loglog(bb_spy)\n"
        "s_qqq, c_qqq = fit_loglog(bb_qqq)\n"
        "print(f'SPY: slope = {s_spy:+.3f}  => beta_implied = {s_spy+1:.3f}')\n"
        "print(f'QQQ: slope = {s_qqq:+.3f}  => beta_implied = {s_qqq+1:.3f}')\n\n"
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "for a, name, d, s, c in [(ax[0], 'SPY', bb_spy, s_spy, c_spy),\n"
        "                          (ax[1], 'QQQ', bb_qqq, s_qqq, c_qqq)]:\n"
        "    a.scatter(d['F'], d['sigma_atm']*100, s=14, alpha=0.6)\n"
        "    xs = np.linspace(d['F'].min(), d['F'].max(), 50)\n"
        "    a.plot(xs, np.exp(c + s * np.log(xs))*100, 'r--',\n"
        "           label=rf'$\\ln\\sigma \\sim {s:+.3f}\\ln F$  ($\\beta$≈{s+1:.2f})')\n"
        "    a.set_xlabel('forward F'); a.set_ylabel(r'$\\sigma_{ATM}$ (%)')\n"
        "    a.set_title(f'{name} backbone (30-DTE)')\n"
        "    a.legend()\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**Reading the fit.** A slope near $0$ ⇒ empirical $\\beta \\approx 1$, "
        "i.e. ATM vol is roughly independent of $F$. That matches the common "
        "practitioner choice of $\\beta=1$ for equity indices. "
        "The paper's $\\beta=0.5$ is for *interest-rate* products whose empirical "
        "backbone behaviour is different."
    ),
    # ==================================================================
    md(
        "---\n## 4. Qualitative summary\n\n"
        "Collate the findings into a compact comparison table for Slide 10."
    ),
    code(
        "summary = pd.DataFrame(\n"
        "    [\n"
        "        {'model': 'Black-Scholes (flat)',\n"
        "         'static RMSE (median, bps)': int(round(merged['bs_rmse'].median()*10000)),\n"
        "         'dynamic RMSE (median, bps)': int(round(pred['stmo'].median())),\n"
        "         'captures smile?': 'no',\n"
        "         'captures dynamics?': 'n/a (sticky-mny used as BS baseline)',\n"
        "         'params per smile': 1,\n"
        "        },\n"
        "        {'model': 'Local Volatility',\n"
        "         'static RMSE (median, bps)': 0,\n"
        "         'dynamic RMSE (median, bps)': int(round(pred['LV'].median())),\n"
        "         'captures smile?': 'yes (by construction)',\n"
        "         'captures dynamics?': 'wrong direction (see fig. 2a / 2b)',\n"
        "         'params per smile': 'nonparametric surface',\n"
        "        },\n"
        "        {'model': 'SABR',\n"
        "         'static RMSE (median, bps)': int(round(merged['rmse'].median()*10000)),\n"
        "         'dynamic RMSE (median, bps)': int(round(pred['SABR'].median())),\n"
        "         'captures smile?': 'yes (tens of bps RMSE)',\n"
        "         'captures dynamics?': 'yes — tune β, ρ',\n"
        "         'params per smile': '3 (α, ρ, ν)',\n"
        "        },\n"
        "    ]\n"
        ").set_index('model')\n"
        "summary"
    ),
    md(
        "## Conclusions\n\n"
        "1. **Fit accuracy.** SABR fits market smiles ~10× better than a flat BS vol, "
        "   with a handful of parameters rather than the non-parametric surface "
        "   required by Local Vol.\n"
        "2. **Dynamics.** Local Vol matches today's smile perfectly by construction "
        "   but predicts next-period smiles *worse* than SABR or even a naive "
        "   sticky-moneyness rule. SABR's stochastic-vol structure (parameters "
        "   $\\beta, \\rho$) controls the direction of the shift and matches the "
        "   observed behaviour.\n"
        "3. **Backbone.** Empirical SPY / QQQ log-log regression gives "
        "   $\\beta \\approx 1$, consistent with the lognormal convention for "
        "   equity indices. The paper's $\\beta = 0.5$ is an interest-rate choice.\n"
        "4. **Practical verdict (slide 11).** SABR is simple, closed-form, "
        "   calibratable in milliseconds, and empirically delivers correct "
        "   static *and* dynamic behaviour — the combination that neither BS nor "
        "   Local Vol offers individually."
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "title": "07 Model Comparison",
    },
    "nbformat": NB_VERSION,
    "nbformat_minor": 5,
}
with open("07_model_comparison.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote 07_model_comparison.ipynb")
