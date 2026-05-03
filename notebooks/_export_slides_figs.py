"""Export presentation-ready figures from the project notebooks.

Output goes to <repo>/figures/ by default, overridable via the
SABR_FIG_DIR environment variable.

Usage:  python3 _export_slides_figs.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# project paths (everything is repo-relative)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from src.sabr import sabr_vol, sabr_vol_atm
from src.sensitivity import (
    atm_slope, atm_curvature, alpha_for_target_atm, backbone,
)
from src.data_loader import load_fred_yields, build_smile
from src.calibration import calibrate_smile_panel
from src.model_compare import (
    fit_flat_bs, local_vol_slice,
    predict_sticky_strike, predict_sticky_moneyness,
    predict_local_vol, predict_sabr, rmse,
)

OUT = os.environ.get("SABR_FIG_DIR", os.path.join(ROOT, "figures"))
os.makedirs(OUT, exist_ok=True)
print(f"writing to {OUT}")

# -------- styling for slides ------------------------------------
plt.rcParams.update({
    "figure.figsize": (8.5, 5.0),
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {name}")


# ================================================================
# Load market data once
# ================================================================
DATA_DIR = os.path.join(ROOT, "Data")
CACHE_DIR = os.path.join(ROOT, "cache")
rates = load_fred_yields(DATA_DIR).ffill()
spy = pd.read_parquet(os.path.join(CACHE_DIR, "spy_options_filtered.parquet"))


# ================================================================
# fig01_smile_motivation.png   (Slide 3)
#   "BS says σ const, market shows smile" — real SPY smile
# ================================================================
sm = build_smile(spy, rates, "2022-01-03", dte=30)
F = sm.attrs["F"]
fig, ax = plt.subplots(figsize=(8.5, 5.0))
mask_p = sm["OPTION"] == "P"
mask_c = sm["OPTION"] == "C"
ax.scatter(sm.loc[mask_p, "STRIKE"], sm.loc[mask_p, "SIGMA_MKT"] * 100,
           color="C3", s=35, label="OTM puts", zorder=3)
ax.scatter(sm.loc[mask_c, "STRIKE"], sm.loc[mask_c, "SIGMA_MKT"] * 100,
           color="C0", s=35, label="OTM calls", zorder=3)
ax.plot(sm["STRIKE"], sm["SIGMA_MKT"] * 100, "k-", lw=1, alpha=0.4)
ax.axhline(sm["SIGMA_MKT"].iloc[(sm["STRIKE"] - F).abs().idxmin()] * 100,
           color="grey", ls="--", lw=1.0, label="ATM (BS would assume flat)")
ax.axvline(F, color="k", ls=":", lw=0.8, label=f"F = {F:.1f}")
ax.set_xlabel("Strike $K$")
ax.set_ylabel("Implied volatility (%)")
ax.set_title("SPY 30-DTE smile — 2022-01-03   (market $\\neq$ flat)")
ax.legend(loc="upper right")
save(fig, "fig01_smile_motivation.png")


# ================================================================
# fig02_local_vol_wrong.png    (Slide 4)
#  Synthetic F-shift demo: SABR vs Local-Vol vs sticky rules
# ================================================================
sm0 = build_smile(spy, rates, "2022-01-03", dte=30)
F0 = sm0.attrs["F"]
T0 = sm0.attrs["expire_dte"] / 365.25
fit0 = calibrate_smile_panel(sm0, beta=0.5)
alpha0, beta0, rho0, nu0 = fit0.alpha, 0.5, fit0.rho, fit0.nu
K_today = sm0["STRIKE"].to_numpy()
s_today = sm0["SIGMA_MKT"].to_numpy()
K_lv = np.linspace(0.6 * F0, 1.4 * F0, 181)
sloc = local_vol_slice(K_lv, F0, T0, alpha0, beta0, rho0, nu0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (F_new, tag) in zip(axes,
                            [(F0 * 0.95, "Forward $F$ falls 5%"),
                             (F0 * 1.05, "Forward $F$ rises 5%")]):
    K_new = np.linspace(0.85 * F_new, 1.15 * F_new, 80)
    s_sabr = predict_sabr(K_new, F_new, T0, alpha0, beta0, rho0, nu0)
    s_stmo = predict_sticky_moneyness(K_new, F_new, K_today, F0, s_today)
    s_lv = predict_local_vol(K_new, F_new, T0, sloc, K_lv)
    ax.plot(K_today, s_today * 100, "k-", lw=2, alpha=0.4, label="today")
    ax.plot(K_new, s_sabr * 100, "C0", lw=2.2, label="SABR")
    ax.plot(K_new, s_stmo * 100, "C2--", lw=1.7, label="market (sticky-mny)")
    ax.plot(K_new, s_lv * 100, "C3--", lw=1.7, label="local vol")
    ax.axvline(F0, color="grey", ls=":", lw=0.7)
    ax.axvline(F_new, color="red", ls=":", lw=0.7)
    ax.set_title(tag)
    ax.set_xlabel("Strike $K$")
    ax.set_ylabel("Implied vol (%)")
    ax.legend(fontsize=10, loc="upper right")
plt.tight_layout()
save(fig, "fig02_local_vol_wrong.png")


# ================================================================
# fig03_baseline_smile.png    (Slide 6 / 7 illustration)
#  Pure SABR baseline smile
# ================================================================
F_, T_ = 100.0, 1.0
alpha, beta, rho, nu = 0.20, 0.5, -0.30, 0.40
Ks = np.linspace(70, 130, 200)
ivs = sabr_vol(Ks, F_, T_, alpha, beta, rho, nu)
fig, ax = plt.subplots()
ax.plot(Ks, ivs * 100, lw=2.2, color="C0")
ax.axvline(F_, color="k", ls="--", lw=0.8, label=f"$F={F_:.0f}$")
ax.set_xlabel("Strike $K$")
ax.set_ylabel("$\\sigma_{\\rm SABR}$ (%)")
ax.set_title(rf"SABR smile  ($\alpha$={alpha}, $\beta$={beta}, $\rho$={rho}, $\nu$={nu}, $T$={T_:.0f})")
ax.legend()
save(fig, "fig03_baseline_smile.png")


# ================================================================
# fig04_sensitivity_alpha.png  (Slide 9) — alpha level
# ================================================================
fig, ax = plt.subplots()
for a in [0.10, 0.15, 0.20, 0.25, 0.30]:
    ax.plot(Ks, sabr_vol(Ks, F_, T_, a, beta, rho, nu) * 100,
            label=rf"$\alpha$={a}", lw=1.8)
ax.axvline(F_, color="k", ls="--", lw=0.6)
ax.set_xlabel("Strike $K$")
ax.set_ylabel("$\\sigma_{\\rm SABR}$ (%)")
ax.set_title(r"Sensitivity to $\alpha$  —  level")
ax.legend()
save(fig, "fig04_sensitivity_alpha.png")


# ================================================================
# fig05_sensitivity_beta.png   (Slide 9) — beta backbone
#   Two-panel:  static smile shape  +  backbone trace
# ================================================================
target_atm = 0.20
alphas = {b: alpha_for_target_atm(F_, T_, b, rho, nu, target_atm)
          for b in (0.0, 0.5, 1.0)}
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for b in (0.0, 0.5, 1.0):
    ax[0].plot(Ks, sabr_vol(Ks, F_, T_, alphas[b], b, rho, nu) * 100,
               lw=1.8, label=rf"$\beta$={b}")
ax[0].axvline(F_, color="k", ls="--", lw=0.6)
ax[0].set_xlabel("Strike $K$")
ax[0].set_ylabel("$\\sigma_{\\rm SABR}$ (%)")
ax[0].set_title(r"Smile shape (matched ATM)")
ax[0].legend()

F_grid = np.linspace(70, 130, 121)
for b in (0.0, 0.5, 1.0):
    bb = backbone(F_grid, T_, alphas[b], b, rho, nu)
    ax[1].plot(F_grid, bb * 100, lw=1.8, label=rf"$\beta$={b}")
ax[1].axvline(F_, color="k", ls="--", lw=0.6)
ax[1].axhline(target_atm * 100, color="grey", ls=":", lw=0.6)
ax[1].set_xlabel("Forward $F$")
ax[1].set_ylabel(r"$\sigma_{ATM}$ (%)")
ax[1].set_title("Backbone:  $\\sigma_{ATM}$ vs $F$")
ax[1].legend()
plt.tight_layout()
save(fig, "fig05_sensitivity_beta.png")


# ================================================================
# fig06_sensitivity_rho.png    (Slide 9) — rho skew
# ================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for r_ in [-0.7, -0.4, -0.1, 0.1, 0.4, 0.7]:
    ax[0].plot(Ks, sabr_vol(Ks, F_, T_, alpha, beta, r_, nu) * 100,
               lw=1.6, label=rf"$\rho$={r_}")
ax[0].axvline(F_, color="k", ls="--", lw=0.6)
ax[0].set_xlabel("Strike $K$")
ax[0].set_ylabel("$\\sigma_{\\rm SABR}$ (%)")
ax[0].set_title(r"Smile shape vs $\rho$")
ax[0].legend(fontsize=10)

rho_axis = np.linspace(-0.8, 0.8, 33)
slopes = [atm_slope(F_, T_, alpha, beta, r_, nu) for r_ in rho_axis]
ax[1].plot(rho_axis, slopes, lw=2, color="C3")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].set_xlabel(r"$\rho$")
ax[1].set_ylabel(r"ATM slope $\partial_{\ln K}\sigma|_{K=F}$")
ax[1].set_title(r"Skew is monotone in $\rho$")
plt.tight_layout()
save(fig, "fig06_sensitivity_rho.png")


# ================================================================
# fig07_sensitivity_nu.png     (Slide 9) — nu curvature
# ================================================================
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for n_ in [0.05, 0.20, 0.40, 0.60, 0.80, 1.00]:
    ax[0].plot(Ks, sabr_vol(Ks, F_, T_, alpha, beta, rho, n_) * 100,
               lw=1.6, label=rf"$\nu$={n_}")
ax[0].axvline(F_, color="k", ls="--", lw=0.6)
ax[0].set_xlabel("Strike $K$")
ax[0].set_ylabel("$\\sigma_{\\rm SABR}$ (%)")
ax[0].set_title(r"Smile shape vs $\nu$")
ax[0].legend(fontsize=10)

nu_axis = np.linspace(0.05, 1.0, 30)
curvs = [atm_curvature(F_, T_, alpha, beta, rho, n_) for n_ in nu_axis]
ax[1].plot(nu_axis, curvs, lw=2, color="C2")
ax[1].set_xlabel(r"$\nu$")
ax[1].set_ylabel(r"ATM curvature $\partial^2_{\ln K}\sigma|_{K=F}$")
ax[1].set_title(r"Smile curvature grows with $\nu$")
plt.tight_layout()
save(fig, "fig07_sensitivity_nu.png")


# ================================================================
# fig08_calibration_fit.png   (Slide 9) — real SPY fit
# ================================================================
sm = build_smile(spy, rates, "2022-01-03", dte=30)
F = sm.attrs["F"]
T = sm.attrs["expire_dte"] / 365.25
K_fine = np.linspace(sm["STRIKE"].min(), sm["STRIKE"].max(), 200)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.scatter(sm["STRIKE"], sm["SIGMA_MKT"] * 100, color="k", s=24,
           label="market", zorder=3)
for b, c in zip((0.0, 0.5, 1.0), ("C0", "C1", "C2")):
    f = calibrate_smile_panel(sm, beta=b)
    ax.plot(K_fine,
            sabr_vol(K_fine, F, T, f.alpha, b, f.rho, f.nu) * 100,
            color=c, lw=1.8,
            label=rf"$\beta$={b}, RMSE={f.rmse*100:.2f}%")
ax.axvline(F, color="grey", ls=":", lw=0.8)
ax.set_xlabel("Strike $K$")
ax.set_ylabel("Implied vol (%)")
ax.set_title(f"SABR fit to SPY 2022-01-03 30-DTE   ($F$={F:.1f})")
ax.legend()
save(fig, "fig08_calibration_fit.png")


# ================================================================
# fig09_rmse_comparison.png   (Slide 10) — SABR vs BS RMSE distribution
# ================================================================
qqq = pd.read_parquet(os.path.join(CACHE_DIR, "qqq_options_filtered.parquet"))
grid = pd.read_parquet(os.path.join(CACHE_DIR, "calibration_grid.parquet"))
cal = pd.read_parquet(os.path.join(CACHE_DIR, "calibration_results.parquet"))

rows = []
by_tk = {"spy": spy, "qqq": qqq}
for _, g in grid.iterrows():
    try:
        sm = build_smile(by_tk[g.ticker], rates, g.trade_date, dte=g.dte)
    except Exception:
        continue
    if len(sm) < 5:
        continue
    sigma_bs, rmse_bs = fit_flat_bs(sm["STRIKE"].to_numpy(),
                                    sm["SIGMA_MKT"].to_numpy())
    rows.append({
        "ticker": g.ticker.upper(),
        "trade_date": g.trade_date,
        "dte": g.dte,
        "bs_rmse": rmse_bs,
    })
bs_df = pd.DataFrame(rows)
merged = bs_df.merge(cal.query("beta == 0.5"),
                     on=["ticker", "trade_date", "dte"])
ratio = float((merged["bs_rmse"] / merged["rmse"]).median())

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].scatter(merged["bs_rmse"] * 10000, merged["rmse"] * 10000, s=18, alpha=0.6)
lim = max(merged["bs_rmse"].max(), merged["rmse"].max()) * 10000
ax[0].plot([0, lim], [0, lim], "k--", lw=0.7)
ax[0].set_xlabel("BS flat RMSE (bps)")
ax[0].set_ylabel(r"SABR ($\beta$=0.5) RMSE (bps)")
ax[0].set_title("Per-smile fit quality (210 smiles)")

data = [merged["rmse"] * 10000, merged["bs_rmse"] * 10000]
ax[1].boxplot(data, tick_labels=["SABR", "BS flat"])
ax[1].set_yscale("log")
ax[1].set_ylabel("RMSE (bps of vol)")
ax[1].set_title(f"BS / SABR median ratio = {ratio:.0f}$\\times$")
plt.tight_layout()
save(fig, "fig09_rmse_comparison.png")


# ================================================================
# fig10_dynamics_test.png     (Slide 10) — empirical next-week prediction
# ================================================================
def empirical_prediction_errors(d1, d2, dte=30):
    sm1 = build_smile(spy, rates, d1, dte=dte)
    sm2 = build_smile(spy, rates, d2, dte=dte)
    F1 = sm1.attrs["F"]
    F2 = sm2.attrs["F"]
    T1 = sm1.attrs["expire_dte"] / 365.25
    T2 = sm2.attrs["expire_dte"] / 365.25
    fit = calibrate_smile_panel(sm1, beta=0.5)
    alpha, rho, nu = fit.alpha, fit.rho, fit.nu
    K_lv = np.linspace(0.6 * F1, 1.4 * F1, 161)
    sloc = local_vol_slice(K_lv, F1, T1, alpha, 0.5, rho, nu)
    K2 = sm2["STRIKE"].to_numpy()
    s2 = sm2["SIGMA_MKT"].to_numpy()
    p_sabr = predict_sabr(K2, F2, T2, alpha, 0.5, rho, nu)
    p_stst = predict_sticky_strike(K2, sm1["STRIKE"].to_numpy(),
                                    sm1["SIGMA_MKT"].to_numpy())
    p_stmo = predict_sticky_moneyness(K2, F2, sm1["STRIKE"].to_numpy(),
                                       F1, sm1["SIGMA_MKT"].to_numpy())
    p_lv = predict_local_vol(K2, F2, T1, sloc, K_lv)
    return {
        "SABR":  rmse(p_sabr, s2),
        "stst":  rmse(p_stst, s2),
        "stmo":  rmse(p_stmo, s2),
        "LV":    rmse(p_lv,   s2),
    }

pair_starts = ["2018-02-02", "2020-03-13", "2022-01-21",
               "2022-06-10", "2023-03-10"]
spy_dates = np.sort(spy["QUOTE_DATE"].unique())

def near_future(d, days=7):
    target = np.datetime64(pd.Timestamp(d) + pd.Timedelta(days=days))
    fut = spy_dates[spy_dates >= target]
    return pd.Timestamp(fut[0]) if len(fut) else None

pairs = [(pd.Timestamp(d), near_future(d)) for d in pair_starts]
errs = pd.DataFrame([empirical_prediction_errors(d1, d2) for d1, d2 in pairs])
medians = errs.median()

fig, ax = plt.subplots(figsize=(8.5, 5.0))
labels = ["SABR", "sticky-strike", "sticky-mny", "local vol"]
values = [medians["SABR"] * 1e4, medians["stst"] * 1e4,
          medians["stmo"] * 1e4, medians["LV"] * 1e4]
bars = ax.bar(labels, values, color=["C0", "C1", "C2", "C3"])
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v:.0f}",
            ha="center", va="bottom", fontsize=12)
ax.set_ylabel("median next-week RMSE (bps of vol)")
ax.set_title(f"Empirical dynamics test ({len(pairs)} date pairs, SPY 30-DTE)")
save(fig, "fig10_dynamics_test.png")


print("\nAll figures written to", OUT)
