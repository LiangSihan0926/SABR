"""Export the additional figures needed for the round-2 presentation.

Reuses fig01-fig10 produced for round 1 in /Users/leungsun/Downloads/figs/
and writes 7 new figures (fig11-fig17) to the same folder.

Run:  python3 export_figs_round2.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- locate project ----
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
DATA_DIR  = os.path.join(ROOT, "Data")
CACHE_DIR = os.path.join(ROOT, "cache")
OUT_DIR   = "/Users/leungsun/Downloads/figs"
os.makedirs(OUT_DIR, exist_ok=True)

from src.sabr import sabr_vol, sabr_vol_atm
from src.sensitivity import (atm_level, atm_slope, atm_curvature,
                              alpha_for_target_atm)
from src.data_loader import load_fred_yields, build_smile, FRED_TENORS
from src.calibration import calibrate_sabr, calibrate_smile_panel

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
})


def save(name):
    p = os.path.join(OUT_DIR, name)
    plt.savefig(p)
    print(f"  -> {p}")
    plt.close()


# load shared data once
print("loading market data...")
rates = load_fred_yields(DATA_DIR).ffill()
spy   = pd.read_parquet(os.path.join(CACHE_DIR, "spy_options_filtered.parquet"))
qqq   = pd.read_parquet(os.path.join(CACHE_DIR, "qqq_options_filtered.parquet"))
grid  = pd.read_parquet(os.path.join(CACHE_DIR, "calibration_grid.parquet"))
cal   = pd.read_parquet(os.path.join(CACHE_DIR, "calibration_results.parquet"))


# ============================================================
# fig11_validation.png
#   Recovery test + ATM consistency error vs beta + put-call parity
# ============================================================
print("fig11_validation ...")
fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

# (a) recovery error scatter for beta=0.5 across noise levels
F, T = 100.0, 1.0
alpha_t, beta_t, rho_t, nu_t = 0.20, 0.5, -0.30, 0.40
K = np.linspace(70, 130, 31)
s_clean = sabr_vol(K, F, T, alpha_t, beta_t, rho_t, nu_t)
noise_levels = np.array([0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3])
errs = []
rng = np.random.default_rng(0)
for nl in noise_levels:
    s_n = s_clean + nl * rng.standard_normal(s_clean.size)
    r = calibrate_sabr(K, s_n, F, T, beta=beta_t)
    errs.append((abs(r.alpha-alpha_t), abs(r.rho-rho_t), abs(r.nu-nu_t)))
errs = np.array(errs)
axes[0].plot(noise_levels*1e4, errs[:, 0], "o-", label=r"$|\Delta\alpha|$")
axes[0].plot(noise_levels*1e4, errs[:, 1], "s-", label=r"$|\Delta\rho|$")
axes[0].plot(noise_levels*1e4, errs[:, 2], "^-", label=r"$|\Delta\nu|$")
axes[0].set_xscale("symlog", linthresh=1)
axes[0].set_yscale("log")
axes[0].set_xlabel("noise level (bps of vol)")
axes[0].set_ylabel("parameter error")
axes[0].set_title("Recovery test on synthetic smile")
axes[0].legend(fontsize=8)

# (b) Smoothness across the ATM point: SABR(K) for K -> F
#     The 2.17 formula has a removable singularity at K=F (z/x(z) -> 1).
#     We branch to (2.18) at the ATM tolerance to keep the curve smooth.
eps_grid = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12])
sigmas_eps = []
for eps in eps_grid:
    Kp = F * (1 + eps)
    sigmas_eps.append(float(sabr_vol(np.array([Kp]), F, T,
                                     0.2, 0.5, rho_t, nu_t)[0]))
sig_atm = sabr_vol_atm(F, T, 0.2, 0.5, rho_t, nu_t)
diffs = np.abs(np.array(sigmas_eps) - sig_atm)
axes[1].loglog(eps_grid, diffs, "o-", label="formula (2.17) at $K=F(1+\\varepsilon)$")
axes[1].axhline(1e-16, color="r", ls=":", label="machine eps")
axes[1].invert_xaxis()
axes[1].set_xlabel(r"$\varepsilon = K/F - 1$")
axes[1].set_ylabel(r"$|\sigma(K) - \sigma_{ATM}|$")
axes[1].set_title("ATM smoothness  (2.17 $\\to$ 2.18 limit)")
axes[1].legend(fontsize=8)

# (c) Put-call parity error across strike
from src.black import black_call, black_put
F_, K_, T_, r_ = 100.0, np.linspace(80, 120, 21), 1.0, 0.05
sigma_ = 0.25
C = black_call(F_, K_, sigma_, T_, r_)
P = black_put (F_, K_, sigma_, T_, r_)
parity_err = (C - P) - np.exp(-r_*T_) * (F_ - K_)
axes[2].plot(K_, np.abs(parity_err), "o-")
axes[2].set_yscale("log")
axes[2].axhline(1e-13, color="r", ls=":", label="machine eps")
axes[2].set_xlabel("K")
axes[2].set_ylabel(r"$|(C-P) - D(F-K)|$")
axes[2].set_title("Put-call parity error")
axes[2].legend(fontsize=8)

plt.tight_layout()
save("fig11_validation.png")


# ============================================================
# fig12_data_overview.png
#   FRED yields + SPY underlying price 2014-2023
# ============================================================
print("fig12_data_overview ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for col in rates.columns:
    axes[0].plot(rates.index, rates[col]*100, label=col, lw=1)
axes[0].set_ylabel("yield (%)")
axes[0].set_title("FRED CMT Treasury yields")
axes[0].xaxis.set_major_locator(mdates.YearLocator(2))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[0].legend(loc="upper left", fontsize=8)

spot = spy.groupby("QUOTE_DATE")["UNDERLYING_LAST"].mean().sort_index()
axes[1].plot(spot.index, spot.values, lw=1, color="C3")
axes[1].set_title("SPY spot (from optionsdx 2014-2023)")
axes[1].set_ylabel("spot")
axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
save("fig12_data_overview.png")


# ============================================================
# fig13_smile_construction.png
#   SPY 2022-01-03 smile, OTM tagged + F vertical line
# ============================================================
print("fig13_smile_construction ...")
sm = build_smile(spy, rates, "2022-01-03", dte=30)
F_, T_, r_ = sm.attrs["F"], sm.attrs["expire_dte"]/365.25, sm.attrs["r"]
fig, ax = plt.subplots(figsize=(8.5, 4.5))
mask_p = sm["OPTION"] == "P"
mask_c = sm["OPTION"] == "C"
ax.scatter(sm.loc[mask_p, "STRIKE"], sm.loc[mask_p, "SIGMA_MKT"]*100,
           color="C3", s=22, label="OTM puts ($K<F$)")
ax.scatter(sm.loc[mask_c, "STRIKE"], sm.loc[mask_c, "SIGMA_MKT"]*100,
           color="C0", s=22, label="OTM calls ($K>F$)")
ax.axvline(F_, color="k", ls="--", lw=1, label=f"$F$ = {F_:.2f}")
ax.set_xlabel("Strike $K$")
ax.set_ylabel("Implied vol (%)")
ax.set_title(f"SPY 2022-01-03 30-DTE smile  -  53 strikes,  $r$ = {r_*100:.2f}%")
ax.legend(loc="upper right")
ax.text(0.02, 0.95,
        r"$F$ extracted via put-call parity at the strike of"
        + "\nminimum $|C_{mid} - P_{mid}|$ (Hagan §3, eq. 2.4)",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
plt.tight_layout()
save("fig13_smile_construction.png")


# ============================================================
# fig14_joint_heatmap.png
#   (rho, nu) -> (ATM slope, ATM curvature)
# ============================================================
print("fig14_joint_heatmap ...")
F, T = 100.0, 1.0
alpha, beta = 0.20, 0.5
rho_ax = np.linspace(-0.8, 0.8, 33)
nu_ax  = np.linspace(0.05, 1.0, 20)
RR, NN = np.meshgrid(rho_ax, nu_ax)
SLOPE = np.vectorize(lambda r, n: atm_slope(F, T, alpha, beta, r, n))(RR, NN)
CURV  = np.vectorize(lambda r, n: atm_curvature(F, T, alpha, beta, r, n))(RR, NN)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
im0 = ax[0].pcolormesh(rho_ax, nu_ax, SLOPE, shading="auto", cmap="RdBu_r",
                        vmin=-abs(SLOPE).max(), vmax=abs(SLOPE).max())
ax[0].set_xlabel(r"$\rho$"); ax[0].set_ylabel(r"$\nu$")
ax[0].set_title(r"ATM slope (skew)  -  driven by $\rho$")
plt.colorbar(im0, ax=ax[0])
im1 = ax[1].pcolormesh(rho_ax, nu_ax, CURV, shading="auto", cmap="viridis")
ax[1].set_xlabel(r"$\rho$"); ax[1].set_ylabel(r"$\nu$")
ax[1].set_title(r"ATM curvature (smile)  -  driven by $\nu$")
plt.colorbar(im1, ax=ax[1])
plt.tight_layout()
save("fig14_joint_heatmap.png")


# ============================================================
# fig15_calibration_recovery.png
#   Synthetic SABR -> calibrate -> overlay
# ============================================================
print("fig15_calibration_recovery ...")
F, T = 100.0, 1.0
alpha_t, beta_t, rho_t, nu_t = 0.20, 0.5, -0.30, 0.40
K = np.linspace(75, 125, 41)
s_clean = sabr_vol(K, F, T, alpha_t, beta_t, rho_t, nu_t)
rng = np.random.default_rng(42)
s_noisy = s_clean + 0.005 * rng.standard_normal(s_clean.size)
r_clean = calibrate_sabr(K, s_clean, F, T, beta=beta_t)
r_noisy = calibrate_sabr(K, s_noisy, F, T, beta=beta_t)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
Kf = np.linspace(K.min(), K.max(), 200)
ax[0].plot(K, s_clean*100, "ko", mfc="none", label="synthetic 'market' (no noise)")
ax[0].plot(Kf, sabr_vol(Kf, F, T, r_clean.alpha, beta_t, r_clean.rho, r_clean.nu)*100,
           "C0", lw=2, label="SABR fit")
ax[0].set_xlabel("K"); ax[0].set_ylabel("IV (%)")
ax[0].set_title(rf"Recovery, no noise:  RMSE = {r_clean.rmse:.1e}")
ax[0].legend()

ax[1].plot(K, s_noisy*100, "ko", mfc="none", label="market + 50 bps noise")
ax[1].plot(Kf, sabr_vol(Kf, F, T, r_noisy.alpha, beta_t, r_noisy.rho, r_noisy.nu)*100,
           "C0", lw=2,
           label=rf"fit  $\alpha$={r_noisy.alpha:.3f}  $\rho$={r_noisy.rho:+.3f}  $\nu$={r_noisy.nu:.3f}")
ax[1].plot(Kf, sabr_vol(Kf, F, T, alpha_t, beta_t, rho_t, nu_t)*100,
           "C2--", lw=1, label=rf"truth  $\alpha$={alpha_t}  $\rho$={rho_t:+}  $\nu$={nu_t}")
ax[1].set_xlabel("K"); ax[1].set_ylabel("IV (%)")
ax[1].set_title(rf"Robustness, 50 bps noise:  RMSE = {r_noisy.rmse*100:.2f}%")
ax[1].legend(fontsize=8)
plt.tight_layout()
save("fig15_calibration_recovery.png")


# ============================================================
# fig16_calibration_timeseries.png
#   alpha, rho, nu over 2014-2023 (SPY 30-DTE, three betas)
# ============================================================
print("fig16_calibration_timeseries ...")
sub = cal.query("ticker == 'SPY' and dte == 30").copy()
sub["trade_date"] = pd.to_datetime(sub["trade_date"])
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for b, color in zip((0.0, 0.5, 1.0), ("C0", "C1", "C2")):
    s = sub.query("beta == @b").sort_values("trade_date")
    axes[0].plot(s["trade_date"], s["alpha"], label=rf"$\beta$={b}", color=color, lw=1)
    axes[1].plot(s["trade_date"], s["rho"],   label=rf"$\beta$={b}", color=color, lw=1)
    axes[2].plot(s["trade_date"], s["nu"],    label=rf"$\beta$={b}", color=color, lw=1)
axes[0].set_ylabel(r"$\alpha$"); axes[0].set_yscale("log")
axes[0].set_title("SPY 30-DTE  -  fitted SABR parameter time series, 2014-2023")
axes[0].legend(loc="upper left", fontsize=9)
axes[1].set_ylabel(r"$\rho$"); axes[1].axhline(0, color="k", lw=0.5)
axes[2].set_ylabel(r"$\nu$")
for ax in axes:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
save("fig16_calibration_timeseries.png")


# ============================================================
# fig17_backbone_empirical.png
#   Empirical log-log regression sigma_ATM vs F  (SPY + QQQ)
# ============================================================
print("fig17_backbone_empirical ...")
def observed_atm(tk, dte):
    df = {"spy": spy, "qqq": qqq}[tk]
    rows = []
    for _, g in grid.query("ticker == @tk and dte == @dte").iterrows():
        try:
            sm = build_smile(df, rates, g.trade_date, dte=dte)
        except Exception:
            continue
        if len(sm) < 5:
            continue
        i = int(np.argmin(np.abs(sm["STRIKE"].to_numpy() - sm.attrs["F"])))
        rows.append({"F": sm.attrs["F"], "sigma_atm": float(sm["SIGMA_MKT"].iloc[i])})
    return pd.DataFrame(rows)


bb_spy = observed_atm("spy", 30)
bb_qqq = observed_atm("qqq", 30)


def fit_loglog(df):
    x = np.log(df["F"].to_numpy())
    y = np.log(df["sigma_atm"].to_numpy())
    s, c = np.polyfit(x, y, 1)
    return s, c


s_spy, c_spy = fit_loglog(bb_spy)
s_qqq, c_qqq = fit_loglog(bb_qqq)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, name, d, s, c in [(ax[0], "SPY", bb_spy, s_spy, c_spy),
                          (ax[1], "QQQ", bb_qqq, s_qqq, c_qqq)]:
    a.scatter(d["F"], d["sigma_atm"]*100, s=18, alpha=0.6)
    xs = np.linspace(d["F"].min(), d["F"].max(), 50)
    a.plot(xs, np.exp(c + s * np.log(xs))*100, "r--",
           label=rf"slope $={s:+.3f}\Rightarrow\beta\approx{s+1:.2f}$")
    a.set_xlabel("forward $F$"); a.set_ylabel(r"$\sigma_{ATM}$ (%)")
    a.set_title(f"{name} 30-DTE empirical backbone")
    a.legend(loc="upper left")
plt.tight_layout()
save("fig17_backbone_empirical.png")

print("done.")
