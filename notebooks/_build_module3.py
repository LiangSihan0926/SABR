"""Builder for Module 3 notebook: data loading + clean-smile preparation.

Run from `notebooks/`:  python3 _build_module3.py
"""
import json

NB_VERSION = 4


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": s}


SETUP = r"""import sys, os, glob, time
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

DATA_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'Data'))
CACHE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', 'cache'))
os.makedirs(CACHE_DIR, exist_ok=True)
print('DATA_DIR :', DATA_DIR)
print('CACHE_DIR:', CACHE_DIR)

from src.data_loader import (
    load_fred_yields, interpolate_yield,
    load_options_file, load_options,
    extract_forward, build_smile, FRED_TENORS,
)
"""


cells = [
    md(
        "# 05 — Data Loading for SABR Calibration\n\n"
        "**Inputs on disk** (already downloaded):\n\n"
        "| Source | Files | Coverage |\n"
        "|---|---|---|\n"
        "| FRED daily Treasury yields | `DGS1MO.csv`, `DGS3MO.csv`, `DGS6MO.csv`, `DGS1.csv`, `DGS2.csv` | 2014-01 → 2023-12 |\n"
        "| SPY EOD options (optionsdx) | `spy_eod_YYYYMM.txt` × 120 | 2014-01 → 2023-12 |\n"
        "| QQQ EOD options (optionsdx) | `qqq_eod_YYYYMM.txt` × 120 | 2014-01 → 2023-12 |\n\n"
        "**Goal.** Produce, for any `(ticker, trade_date, expiry)`, a clean "
        "smile DataFrame `(K, K/F, T, F, r, sigma_mkt, ...)` that Module 4 "
        "can feed into `scipy.optimize.least_squares`.\n\n"
        "**Sub-steps.**\n"
        "1. FRED yield curves — plot and verify linear interpolation.\n"
        "2. Load a single month of SPY options; inspect the filtered frame.\n"
        "3. Build one smile — $F$ via put-call parity, OTM-only filter.\n"
        "4. Load all 10 years of SPY + QQQ; cache to parquet.\n"
        "5. Smile-panel: the same expiry on several dates → sanity check for "
        "   Module 5 (dynamics comparison).\n"
        "6. Export a curated list of calibration dates for Module 4."
    ),
    code(SETUP),
    # -----------------------------------------------------------
    md("---\n## 1. FRED Treasury yield curves"),
    code(
        "rates = load_fred_yields(DATA_DIR)\n"
        "rates = rates.ffill()   # carry yields over US holidays\n"
        "print(f'shape = {rates.shape}')\n"
        "print(f'date range = {rates.index.min().date()} -> {rates.index.max().date()}')\n"
        "rates.head()"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
        "for col in rates.columns:\n"
        "    ax.plot(rates.index, rates[col]*100, label=col, lw=1)\n"
        "ax.set_ylabel('yield (%)')\n"
        "ax.set_title('US Treasury CMT yields (FRED)')\n"
        "ax.xaxis.set_major_locator(mdates.YearLocator())\n"
        "ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
        "ax.legend(loc='upper left'); plt.show()"
    ),
    md(
        "### Yield-curve snapshots — interpolation sanity check\n"
        "Plot the observed (tenor, yield) points on a handful of dates "
        "and the linearly interpolated curve used by `interpolate_yield`."
    ),
    code(
        "import numpy as np\n"
        "taus_fine = np.linspace(1/12, 2.0, 60)\n"
        "sample_dates = ['2016-06-30', '2019-06-28', '2022-01-03', '2023-06-30']\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
        "for d in sample_dates:\n"
        "    row = rates.loc[d].dropna()\n"
        "    tenors = np.array([FRED_TENORS[c] for c in row.index])\n"
        "    yields = row.values * 100\n"
        "    curve = [interpolate_yield(rates, d, t)*100 for t in taus_fine]\n"
        "    p = ax.plot(taus_fine, curve, label=d)[0]\n"
        "    ax.scatter(tenors, yields, color=p.get_color(), zorder=3)\n"
        "ax.set_xlabel('tenor (years)'); ax.set_ylabel('yield (%)')\n"
        "ax.set_title('FRED snapshots + linear interpolation')\n"
        "ax.legend(); plt.show()"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 2. Load one month of SPY options\n\n"
        "`load_options_file` parses the optionsdx format, normalises the "
        "bracketed header, coerces numerics, and applies three filters:\n"
        "* DTE ∈ [5, 365] (drop 0-DTE row at top of file; drop ultra-long)\n"
        "* |STRIKE_DISTANCE_PCT| ≤ 0.30 (keep ±30 % of spot)\n"
        "* strike > 0."
    ),
    code(
        "t0 = time.time()\n"
        "jan22 = load_options_file(os.path.join(DATA_DIR, 'spy', 'spy_eod_202201.txt'))\n"
        "print(f'rows kept: {len(jan22):,}   ({time.time()-t0:.2f}s)')\n"
        "jan22.head()"
    ),
    code(
        "# what trade-dates and DTE values exist?\n"
        "print('trade dates:', jan22['QUOTE_DATE'].dt.date.unique()[:5], '...')\n"
        "print('num trade dates in month:', jan22['QUOTE_DATE'].nunique())\n"
        "print('DTEs present:', sorted(jan22['DTE'].unique())[:15], '...')\n"
        "print('strike coverage (pct of spot):',\n"
        "      f\"{jan22['STRIKE_DISTANCE_PCT'].min():+.3f}\",\n"
        "      f\"{jan22['STRIKE_DISTANCE_PCT'].max():+.3f}\")"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 3. Build one smile — Jan 3 2022, 30-day expiry\n\n"
        "`build_smile` does four things:\n"
        "1. Slice to the chosen `(trade_date, DTE)`.\n"
        "2. Compute $T$ in years, interpolate $r$ from the yield curve.\n"
        "3. Extract the implied forward $F$ via put-call parity at the "
        "   strike with the smallest $|C-P|$.\n"
        "4. Keep only the **OTM side** (puts for $K<F$, calls for $K>F$) "
        "   with bid ≥ 5 ¢ — the liquid part of the smile."
    ),
    code(
        "smile = build_smile(jan22, rates, trade_date='2022-01-03', dte=30)\n"
        "meta = smile.attrs\n"
        "print(f\"F = {meta['F']:.3f},  T = {meta['expire_dte']:.1f} days,  r = {meta['r']*100:.3f}%\")\n"
        "print(f'{len(smile)} strikes retained')\n"
        "smile.head()"
    ),
    code(
        "fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))\n"
        "mask_p = smile['OPTION'] == 'P'\n"
        "mask_c = smile['OPTION'] == 'C'\n"
        "ax[0].plot(smile['STRIKE'], smile['SIGMA_MKT']*100, 'k-', lw=0.8, alpha=0.5)\n"
        "ax[0].scatter(smile.loc[mask_p, 'STRIKE'], smile.loc[mask_p, 'SIGMA_MKT']*100,\n"
        "              color='C3', label='OTM puts')\n"
        "ax[0].scatter(smile.loc[mask_c, 'STRIKE'], smile.loc[mask_c, 'SIGMA_MKT']*100,\n"
        "              color='C0', label='OTM calls')\n"
        "ax[0].axvline(meta['F'], color='k', lw=0.6, ls='--', label=f\"F={meta['F']:.2f}\")\n"
        "ax[0].set_xlabel('Strike K'); ax[0].set_ylabel('Implied vol (%)')\n"
        "ax[0].set_title(f\"SPY smile, trade={smile.attrs['trade_date'].date()}, DTE={meta['expire_dte']:.0f}\")\n"
        "ax[0].legend()\n\n"
        "ax[1].plot(smile['K_over_F'], smile['SIGMA_MKT']*100, 'ko-', lw=0.8, mfc='none')\n"
        "ax[1].axvline(1.0, color='k', lw=0.6, ls='--')\n"
        "ax[1].set_xlabel('K / F'); ax[1].set_ylabel('Implied vol (%)')\n"
        "ax[1].set_title('Same smile, in moneyness')\n"
        "plt.tight_layout(); plt.show()"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 4. Load 10-year history for SPY + QQQ and cache to parquet\n\n"
        "The full optionsdx dump is ~12 M rows per ticker before filtering, "
        "but only 4-5 M after the 30 %-moneyness window.  We cache the "
        "filtered frame to parquet so subsequent notebooks reload in "
        "a few seconds instead of ~1 min."
    ),
    code(
        "def cache_ticker(ticker):\n"
        "    cache = os.path.join(CACHE_DIR, f'{ticker}_options_filtered.parquet')\n"
        "    if os.path.exists(cache):\n"
        "        df = pd.read_parquet(cache)\n"
        "        print(f'[{ticker}] loaded cached parquet  rows={len(df):,}')\n"
        "    else:\n"
        "        t0 = time.time()\n"
        "        df = load_options(ticker, DATA_DIR)\n"
        "        df.to_parquet(cache)\n"
        "        print(f'[{ticker}] built+cached  rows={len(df):,}  ({time.time()-t0:.1f}s)')\n"
        "    return df\n\n"
        "spy = cache_ticker('spy')\n"
        "qqq = cache_ticker('qqq')"
    ),
    code(
        "# quick sanity: coverage per year\n"
        "def cov_by_year(df, name):\n"
        "    g = (df.assign(year=df['QUOTE_DATE'].dt.year)\n"
        "           .groupby('year').agg(\n"
        "                trade_days=('QUOTE_DATE', 'nunique'),\n"
        "                rows=('STRIKE', 'count'),\n"
        "                max_dte=('DTE', 'max'),\n"
        "          ))\n"
        "    g['ticker'] = name\n"
        "    return g\n\n"
        "pd.concat([cov_by_year(spy, 'SPY'), cov_by_year(qqq, 'QQQ')])"
    ),
    code(
        "# SPY underlying price over the full sample\n"
        "daily_spot = (spy.groupby('QUOTE_DATE')['UNDERLYING_LAST']\n"
        "                 .mean().sort_index())\n"
        "fig, ax = plt.subplots(figsize=(10, 4))\n"
        "ax.plot(daily_spot.index, daily_spot.values, lw=1)\n"
        "ax.set_title('SPY spot (average across its options file) 2014-2023')\n"
        "ax.set_ylabel('spot')\n"
        "ax.xaxis.set_major_locator(mdates.YearLocator())\n"
        "ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))\n"
        "plt.show()"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 5. Smile panel across several dates\n\n"
        "Pick the same nominal DTE (30 days) on a grid of trade dates "
        "spanning several volatility regimes (Brexit, Covid, 2022 drawdown, "
        "post-ZIRP). This panel is the key input to **Module 5**: showing "
        "that the smile really does move *with the market* — the motivation "
        "for SABR over Local-Vol."
    ),
    code(
        "panel_dates = ['2016-06-30', '2018-02-05', '2020-03-16',\n"
        "               '2022-01-03', '2022-06-15', '2023-06-30']\n\n"
        "fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)\n"
        "for ax, d in zip(axes.flat, panel_dates):\n"
        "    try:\n"
        "        sm = build_smile(spy, rates, d, dte=30)\n"
        "    except ValueError as e:\n"
        "        ax.set_title(f'{d}: {e}'); continue\n"
        "    ax.plot(sm['K_over_F'], sm['SIGMA_MKT']*100, 'ko-', mfc='none', lw=0.8)\n"
        "    ax.axvline(1.0, color='k', lw=0.6, ls='--')\n"
        "    ax.set_title(f\"{d}  F={sm.attrs['F']:.1f}  DTE={sm.attrs['expire_dte']:.0f}\")\n"
        "    ax.set_xlabel('K / F'); ax.set_ylabel('IV (%)')\n"
        "plt.tight_layout(); plt.show()"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## 6. Curated calibration grid for Module 4\n\n"
        "For the calibration / sensitivity study we will iterate over a "
        "representative cross-section rather than every trade date. "
        "Save a small DataFrame of `(ticker, trade_date, dte)` triples to "
        "the cache folder."
    ),
    code(
        "calib_dates = pd.date_range('2015-01-05', '2023-12-15', freq='QS-JAN')  # quarter starts\n"
        "# move each to the nearest actually-observed trade date\n"
        "trade_days = spy['QUOTE_DATE'].unique()\n"
        "def snap(d):\n"
        "    d = pd.Timestamp(d)\n"
        "    diffs = np.abs(trade_days - np.datetime64(d))\n"
        "    return pd.Timestamp(trade_days[diffs.argmin()])\n"
        "calib_dates = pd.DatetimeIndex([snap(d) for d in calib_dates]).unique()\n\n"
        "calib_grid = pd.DataFrame([\n"
        "    {'ticker': tk, 'trade_date': d, 'dte': dte}\n"
        "    for tk in ('spy', 'qqq')\n"
        "    for d in calib_dates\n"
        "    for dte in (30, 60, 90)\n"
        "])\n"
        "calib_grid.to_parquet(os.path.join(CACHE_DIR, 'calibration_grid.parquet'))\n"
        "print(f'calibration grid: {len(calib_grid)} rows')\n"
        "print(f'  unique dates: {calib_grid[\"trade_date\"].nunique()}')\n"
        "calib_grid.head()"
    ),
    code(
        "# quick check: can we build a smile for every row in the grid?\n"
        "fails = 0; ok = 0; empties = 0\n"
        "by_tk = {'spy': spy, 'qqq': qqq}\n"
        "for _, row in calib_grid.iterrows():\n"
        "    try:\n"
        "        sm = build_smile(by_tk[row.ticker], rates, row.trade_date, dte=row.dte)\n"
        "        if len(sm) < 5:\n"
        "            empties += 1\n"
        "        else:\n"
        "            ok += 1\n"
        "    except Exception:\n"
        "        fails += 1\n"
        "print(f'smile-build results:  ok={ok}   thin(<5 strikes)={empties}   fail={fails}')"
    ),
    # -----------------------------------------------------------
    md(
        "---\n## Summary\n\n"
        "* FRED yield curves loaded, forward-filled across US holidays, "
        "interpolated linearly for any tenor in `[1/12, 2]` years.\n"
        "* optionsdx EOD files parsed, cleaned and filtered; two parquet "
        "caches written:\n"
        "  ```\n"
        "  cache/spy_options_filtered.parquet\n"
        "  cache/qqq_options_filtered.parquet\n"
        "  cache/calibration_grid.parquet\n"
        "  ```\n"
        "* `build_smile(...)` produces a tidy $(K, K/F, T, F, r, \\sigma)$ "
        "frame — exactly what `sabr_vol` and the least-squares calibrator "
        "in Module 4 will consume.\n"
        "* Smile panel across 2016-2023 shows realistic time-variation in "
        "level, skew and curvature that the SABR parameters $(\\alpha, "
        "\\rho, \\nu)$ will have to track.\n\n"
        "→ **Module 4** now has everything it needs: smile observations + "
        "an implementation of $\\sigma_{SABR}(K, F, T; \\alpha, \\beta, "
        "\\rho, \\nu)$."
    ),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "title": "05 Data Loading",
    },
    "nbformat": NB_VERSION,
    "nbformat_minor": 5,
}

with open("05_data_loading.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote 05_data_loading.ipynb")
