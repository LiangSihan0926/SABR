"""Extract numbers for Appendix F (calibration breakdown) and G (dynamics test).

Run with python3.10 from notebooks/.
"""
import os, sys, json, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "Data")
CACHE_DIR = os.path.join(ROOT, "cache")

from src.data_loader import load_fred_yields, build_smile
from src.calibration import calibrate_smile_panel
from src.sabr import sabr_vol
from src.model_compare import (
    fit_flat_bs, local_vol_slice,
    predict_sabr, predict_sticky_strike,
    predict_sticky_moneyness, predict_local_vol, rmse,
)

cal = pd.read_parquet(os.path.join(CACHE_DIR, "calibration_results.parquet"))


# ================================================================
# Appendix F: per (ticker, DTE, beta) RMSE breakdown
# ================================================================
print("=" * 60)
print("APPENDIX F  -  RMSE breakdown by (ticker, DTE, beta), bps of vol")
print("=" * 60)
g = (cal.assign(rmse_bps=cal["rmse"] * 1e4)
        .groupby(["ticker", "dte", "beta"])
        .agg(n=("rmse_bps", "size"),
             med=("rmse_bps", "median"),
             mean=("rmse_bps", "mean"),
             p90=("rmse_bps", lambda s: np.percentile(s, 90)),
             worst=("rmse_bps", "max"),
             nfev_med=("nfev", "median"))
        .round(1))
print(g.to_string())
print()

# ================================================================
# Appendix G: per-pair dynamics test
# ================================================================
print("=" * 60)
print("APPENDIX G  -  Dynamics test, per pair, in bps of vol")
print("=" * 60)
rates = load_fred_yields(DATA_DIR).ffill()
spy = pd.read_parquet(os.path.join(CACHE_DIR, "spy_options_filtered.parquet"))

pair_starts = ["2018-02-02", "2020-03-13", "2022-01-21",
               "2022-06-10", "2023-03-10"]
spy_dates = np.sort(spy["QUOTE_DATE"].unique())


def nearest_future(d, shift_days=7):
    target = np.datetime64(pd.Timestamp(d) + pd.Timedelta(days=shift_days))
    fut = spy_dates[spy_dates >= target]
    return pd.Timestamp(fut[0]) if len(fut) else None


pairs = [(pd.Timestamp(d), nearest_future(d)) for d in pair_starts]


rows = []
for d1, d2 in pairs:
    if d2 is None:
        continue
    sm1 = build_smile(spy, rates, d1, dte=30)
    sm2 = build_smile(spy, rates, d2, dte=30)
    F1, F2 = sm1.attrs["F"], sm2.attrs["F"]
    T1, T2 = sm1.attrs["expire_dte"]/365.25, sm2.attrs["expire_dte"]/365.25
    fit = calibrate_smile_panel(sm1, beta=0.5)
    alpha, rho, nu = fit.alpha, fit.rho, fit.nu

    K_lv = np.linspace(0.6 * F1, 1.4 * F1, 161)
    sloc = local_vol_slice(K_lv, F1, T1, alpha, 0.5, rho, nu)

    K2 = sm2["STRIKE"].to_numpy()
    s2 = sm2["SIGMA_MKT"].to_numpy()

    p_sabr = predict_sabr(K2, F2, T2, alpha, 0.5, rho, nu)
    p_stst = predict_sticky_strike(K2,
                                    sm1["STRIKE"].to_numpy(),
                                    sm1["SIGMA_MKT"].to_numpy())
    p_stmo = predict_sticky_moneyness(K2, F2,
                                       sm1["STRIKE"].to_numpy(),
                                       F1,
                                       sm1["SIGMA_MKT"].to_numpy())
    p_lv = predict_local_vol(K2, F2, T1, sloc, K_lv)

    rows.append({
        "d1": d1.strftime("%Y-%m-%d"),
        "d2": d2.strftime("%Y-%m-%d"),
        "F1": round(F1, 1),
        "F2": round(F2, 1),
        "dF_pct": round((F2/F1 - 1)*100, 2),
        "n_strikes": len(sm2),
        "SABR_bps":   round(rmse(p_sabr, s2) * 1e4, 1),
        "stst_bps":   round(rmse(p_stst, s2) * 1e4, 1),
        "stmo_bps":   round(rmse(p_stmo, s2) * 1e4, 1),
        "LV_bps":     round(rmse(p_lv,   s2) * 1e4, 1),
    })

per_pair = pd.DataFrame(rows)
print(per_pair.to_string(index=False))

# write to a json file so the LaTeX side can ingest if needed
out = {
    "F_breakdown": g.reset_index().to_dict(orient="records"),
    "G_per_pair":  per_pair.to_dict(orient="records"),
    "G_medians":   {
        "SABR":   float(per_pair["SABR_bps"].median()),
        "stst":   float(per_pair["stst_bps"].median()),
        "stmo":   float(per_pair["stmo_bps"].median()),
        "LV":     float(per_pair["LV_bps"].median()),
    }
}
with open(os.path.join(CACHE_DIR, "appendix_data.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nwrote {CACHE_DIR}/appendix_data.json")
