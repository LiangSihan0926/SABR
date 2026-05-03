"""Three additional figures for the Final Project Report.

Reads cache/calibration_results.parquet (1,260 fits) produced by Module 4
and writes:

    fig18_term_structure.png       median (alpha, rho, nu) by DTE
    fig19_spy_qqq_scatter.png      same-date SPY vs QQQ parameter scatter
    fig20_rmse_distribution.png    1,260-fit RMSE histogram, three betas

Run:  python3 export_figs_round3.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_DIR   = "/Users/leungsun/Downloads/figs"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})

cal = pd.read_parquet(os.path.join(CACHE_DIR, "calibration_results.parquet"))
cal["trade_date"] = pd.to_datetime(cal["trade_date"])
print(f"loaded {len(cal):,} fits  ({cal['ticker'].nunique()} tickers, "
      f"{cal['dte'].nunique()} DTEs, {cal['beta'].nunique()} betas)")


# ============================================================
# fig18 — Term structure: median (alpha, rho, nu) by DTE
#         One column per parameter, x-axis = DTE, color = beta,
#         marker = ticker.
# ============================================================
print("fig18_term_structure ...")
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

dtes = sorted(cal["dte"].unique())
betas = sorted(cal["beta"].unique())
beta_colors = {0.0: "C0", 0.5: "C1", 1.0: "C2"}
ticker_marker = {"SPY": "o", "QQQ": "s"}

for ax, param, ylabel in zip(
    axes,
    ["alpha", "rho", "nu"],
    [r"median $\hat\alpha$", r"median $\hat\rho$", r"median $\hat\nu$"],
):
    for tk, mk in ticker_marker.items():
        for b, c in beta_colors.items():
            sub = cal.query("ticker == @tk and beta == @b")
            med = sub.groupby("dte")[param].median()
            ax.plot(
                med.index, med.values,
                marker=mk, color=c, lw=1.3, ms=7,
                mfc=("white" if tk == "QQQ" else c),
                mec=c,
                label=rf"{tk} $\beta$={b}",
            )
    ax.set_xlabel("DTE (days)")
    ax.set_ylabel(ylabel)
    if param == "alpha":
        ax.set_yscale("log")

# clean legend on the third axis
handles, labels = axes[-1].get_legend_handles_labels()
axes[-1].legend(handles, labels, fontsize=7, loc="upper right",
                ncol=2, frameon=True)
axes[0].set_title(r"$\alpha$  -  level (log scale)")
axes[1].set_title(r"$\rho$  -  skew")
axes[2].set_title(r"$\nu$  -  curvature")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig18_term_structure.png"))
plt.close()
print(f"  -> {OUT_DIR}/fig18_term_structure.png")


# ============================================================
# fig19 — SPY vs QQQ parameter scatter (same trade date, DTE=30, beta=0.5)
# ============================================================
print("fig19_spy_qqq_scatter ...")
sub = cal.query("dte == 30 and beta == 0.5").copy()
piv = sub.pivot(index="trade_date", columns="ticker",
                values=["alpha", "rho", "nu", "rmse"])

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

for ax, param, label in zip(
    axes, ["alpha", "rho", "nu"],
    [r"$\hat\alpha$", r"$\hat\rho$", r"$\hat\nu$"],
):
    if param not in piv.columns.get_level_values(0):
        continue
    x = piv[param]["SPY"].dropna()
    y = piv[param]["QQQ"].reindex(x.index).dropna()
    common = x.index.intersection(y.index)
    x = x.loc[common]
    y = y.loc[common]
    ax.scatter(x, y, s=24, alpha=0.65, color="C0", edgecolor="white")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="$y=x$")

    if len(x) > 1:
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{label}    corr$=${corr:+.3f}    ($n={len(x)}$)")
    else:
        ax.set_title(label)
    ax.set_xlabel(f"SPY {label}")
    ax.set_ylabel(f"QQQ {label}")
    ax.legend(fontsize=8, loc="upper left")

axes[0].set_xscale("log")
axes[0].set_yscale("log")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig19_spy_qqq_scatter.png"))
plt.close()
print(f"  -> {OUT_DIR}/fig19_spy_qqq_scatter.png")


# ============================================================
# fig20 — RMSE distribution histogram, three betas overlaid
# ============================================================
print("fig20_rmse_distribution ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

bins = np.linspace(0, 220, 45)  # bps of vol
for b, c in beta_colors.items():
    s = cal.query("beta == @b")["rmse"].to_numpy() * 1e4   # bps
    axes[0].hist(s, bins=bins, color=c, alpha=0.55,
                 label=rf"$\beta$={b}  ($n=${len(s)})", density=False)
axes[0].set_xlabel("Calibration RMSE (bps of vol)")
axes[0].set_ylabel("Number of fits")
axes[0].set_title(f"RMSE distribution across all {len(cal):,} calibrations")
axes[0].legend(fontsize=9)

# right panel: cumulative ECDF for cleaner comparison
for b, c in beta_colors.items():
    s = np.sort(cal.query("beta == @b")["rmse"].to_numpy() * 1e4)
    f = np.arange(1, len(s) + 1) / len(s)
    axes[1].step(s, f, where="post", color=c, lw=1.6,
                 label=rf"$\beta$={b}")
axes[1].set_xlabel("RMSE (bps of vol)")
axes[1].set_ylabel("ECDF")
axes[1].set_title("Empirical CDF, three $\\beta$ regimes")
axes[1].axhline(0.5, color="k", lw=0.5, ls=":")
axes[1].axhline(0.9, color="k", lw=0.5, ls=":")
axes[1].set_xlim(0, 220)
axes[1].legend(fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig20_rmse_distribution.png"))
plt.close()
print(f"  -> {OUT_DIR}/fig20_rmse_distribution.png")


# ============================================================
# Print summary numbers for the paper text
# ============================================================
print("\n=== Summary numbers for paper text ===")
print("\n[Term structure - median values per (ticker, dte, beta=0.5)]")
ts = cal.query("beta == 0.5").groupby(["ticker", "dte"])[
    ["alpha", "rho", "nu", "rmse"]
].median().round(4)
print(ts)

print("\n[SPY vs QQQ correlations, dte=30, beta=0.5]")
for p in ("alpha", "rho", "nu"):
    if p in piv.columns.get_level_values(0):
        x = piv[p]["SPY"].dropna()
        y = piv[p]["QQQ"].reindex(x.index).dropna()
        common = x.index.intersection(y.index)
        if len(common) > 1:
            corr = np.corrcoef(x.loc[common], y.loc[common])[0, 1]
            print(f"  {p}: corr = {corr:+.3f}, n = {len(common)}")

print("\n[RMSE percentiles by beta, in bps of vol]")
for b in betas:
    s = cal.query("beta == @b")["rmse"].to_numpy() * 1e4
    print(f"  beta={b}: median={np.median(s):5.1f}  p90={np.percentile(s, 90):5.1f}  max={s.max():5.1f}")

print("\ndone.")
