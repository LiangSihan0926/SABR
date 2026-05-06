"""Three figures for the EVENT-ANCHORED stress study (the angle
implemented in src/event_stress.py).

These are deliberately distinct from the alpha-bucket regime
classification in src/regime.py and from anything reported in the
reference paper this project compares against. In particular, we
do NOT report a per-regime summary table (the reference does that).
The event-anchored regime tagging is used only as an INTERNAL pipeline
for selecting historical SABR calibrations that feed the portfolio
scenario replay --- the headline deliverable.

    fig23_stress_event_calendar.png      timeline of identified events
    fig24_event_window_trajectory.png    SABR parameter trajectory in
                                         event window (days from event)
    fig25_portfolio_stress_replay.png    portfolio PnL across events,
                                         unhedged vs hedged

Run from notebooks/:  python3 export_figs_round5.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
DATA_DIR  = os.path.join(ROOT, "Data")
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

from src.event_stress import (
    find_stress_events,
    MarketState, hedge_coverage,
)
from src.data_loader import load_fred_yields, build_smile
from src.calibration import calibrate_smile_panel

print("Loading market data and 630 calibrations...")
spy   = pd.read_parquet(os.path.join(CACHE_DIR, "spy_options_filtered.parquet"))
rates = load_fred_yields(DATA_DIR).ffill()


# ============================================================
# Identify events
# ============================================================
events = find_stress_events(spy, return_threshold=-0.03)
print(f"\nFound {len(events)} stress events <= -3% in 2014-2023:")
print(events.head(10))


# ============================================================
# fig23 - Stress event calendar timeline
# ============================================================
print("\nfig23_stress_event_calendar ...")
fig, ax = plt.subplots(figsize=(13, 3.0))
spot = spy.groupby("QUOTE_DATE")["UNDERLYING_LAST"].mean().sort_index()
ax.plot(spot.index, spot.values, color="C0", lw=0.8, alpha=0.7,
        label="SPY spot")
ax.scatter(events.index, events["F_event"],
           color="C3", s=70, zorder=5, edgecolor="black",
           label=f"stress events ($\\Delta F\\leq-3\\%$, $n={len(events)}$)")

worst3 = events.head(3)
for d, row in worst3.iterrows():
    ax.annotate(f"{d.date()}\n{row['return_pct']:+.1f}%",
                xy=(d, row["F_event"]),
                xytext=(8, -32), textcoords="offset points",
                fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="-", color="C3", lw=0.6))

ax.set_ylabel("SPY level")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper left", fontsize=9)
ax.set_title("Historical stress events on SPY, 2014-2023  "
             f"(n={len(events)} days with single-day return $\\leq -3\\%$)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig23_stress_event_calendar.png"))
plt.close()
print(f"  -> {OUT_DIR}/fig23_stress_event_calendar.png")


# ============================================================
# fig24 - SABR parameter trajectory in event window
#         (event-time aligned, x = days from event,
#          y = parameter, with median + IQR band across events)
# ============================================================
print("\nfig24_event_window_trajectory ...")

window_days = 5
spy_dates = pd.DatetimeIndex(spy["QUOTE_DATE"].unique()).normalize().sort_values()


def nearest_trading_day(target, side="next"):
    """Snap to the nearest available trading date in spy_dates.

    side='next' picks the next available date >= target;
    side='prev' picks the last available date <= target.
    """
    if side == "next":
        cands = spy_dates[spy_dates >= target]
        return cands[0] if len(cands) else None
    cands = spy_dates[spy_dates <= target]
    return cands[-1] if len(cands) else None


print("  Calibrating SABR on event-window days (top 8 events x 11 days)...")
trace_rows = []
for ev_date in events.head(8).index:
    for offset in range(-window_days, window_days + 1):
        target = (ev_date + pd.Timedelta(days=offset)).normalize()
        if offset >= 0:
            actual = nearest_trading_day(target, side="next")
        else:
            actual = nearest_trading_day(target, side="prev")
        if actual is None or actual not in set(spy_dates):
            continue
        try:
            sm = build_smile(spy, rates,
                             actual.strftime("%Y-%m-%d"), dte=30)
            fit = calibrate_smile_panel(sm, beta=0.5)
        except Exception:
            continue
        trace_rows.append({
            "event_date":  ev_date,
            "offset_days": offset,
            "actual_date": actual,
            "alpha":       fit.alpha,
            "rho":         fit.rho,
            "nu":          fit.nu,
        })

trace = pd.DataFrame(trace_rows)
print(f"  Collected {len(trace)} per-event-day calibrations.")

agg = trace.groupby("offset_days").agg(
    n        = ("alpha", "size"),
    alpha_med= ("alpha", "median"),
    alpha_q1 = ("alpha", lambda s: s.quantile(0.25)),
    alpha_q3 = ("alpha", lambda s: s.quantile(0.75)),
    rho_med  = ("rho",   "median"),
    rho_q1   = ("rho",   lambda s: s.quantile(0.25)),
    rho_q3   = ("rho",   lambda s: s.quantile(0.75)),
    nu_med   = ("nu",    "median"),
    nu_q1    = ("nu",    lambda s: s.quantile(0.25)),
    nu_q3    = ("nu",    lambda s: s.quantile(0.75)),
).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, p, ylabel in zip(
    axes, ["alpha", "rho", "nu"],
    [r"$\hat\alpha$", r"$\hat\rho$", r"$\hat\nu$"],
):
    med = agg[f"{p}_med"]
    q1  = agg[f"{p}_q1"]
    q3  = agg[f"{p}_q3"]
    ax.fill_between(agg["offset_days"], q1, q3,
                    color="C3", alpha=0.18, label="IQR (25-75th)")
    ax.plot(agg["offset_days"], med, "-o", color="C3",
            lw=1.8, ms=5, label="median across events")
    ax.axvline(0, color="k", lw=0.6, ls="--", label="event day")
    ax.set_xlabel("Days from stress event")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{p.upper()} trajectory in event window")
    ax.legend(fontsize=8, loc="best")
    if p == "alpha":
        ax.set_yscale("log")

plt.suptitle(f"Event-time-aligned SABR parameter trajectories  "
             f"(top 8 events, $\\pm$5 trading days, SPY 30-DTE, $\\beta=0.5$)",
             y=1.03, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig24_event_window_trajectory.png"))
plt.close()
print(f"  -> {OUT_DIR}/fig24_event_window_trajectory.png")


# ============================================================
# fig25 - Portfolio scenario replay (unchanged)
# ============================================================
print("\nfig25_portfolio_stress_replay ...")

today_date = spy["QUOTE_DATE"].max()
today_smile = build_smile(spy, rates, today_date.strftime("%Y-%m-%d"), dte=30)
today_fit = calibrate_smile_panel(today_smile, beta=0.5)
F_today = today_smile.attrs["F"]
T_today = today_smile.attrs["expire_dte"] / 365.25
r_today = today_smile.attrs["r"]

baseline = MarketState(
    F=F_today, T=T_today, r=r_today,
    alpha=today_fit.alpha, beta=0.5,
    rho=today_fit.rho, nu=today_fit.nu,
    label=f"today ({today_date.date()})",
)

K_short = round(F_today * 0.95)
K_hedge = round(F_today * 0.90)
position = [{"strike": K_short, "type": "P", "qty": -100}]
hedge    = [{"strike": K_hedge, "type": "P", "qty": +100}]

print("  Calibrating SABR on each event day for stress states...")
stresses = []
for ev_date, ev_row in events.head(8).iterrows():
    try:
        sm = build_smile(spy, rates, ev_date.strftime("%Y-%m-%d"), dte=30)
    except Exception as e:
        print(f"    skip {ev_date.date()}: {e}")
        continue
    fit = calibrate_smile_panel(sm, beta=0.5)
    shock = ev_row["return_pct"] / 100.0
    F_stressed = F_today * (1.0 + shock)
    stresses.append(MarketState(
        F=F_stressed, T=T_today, r=r_today,
        alpha=fit.alpha, beta=0.5, rho=fit.rho, nu=fit.nu,
        label=f"{ev_date.date()}\n({shock*100:+.1f}%)",
    ))

cov = hedge_coverage(position, hedge, baseline, stresses)
cov = cov.sort_values("return_pct").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(cov))
w = 0.38
ax.bar(x - w/2, cov["unhedged_pnl"]/1000, w,
       color="C3", alpha=0.8, label="unhedged")
ax.bar(x + w/2, cov["hedged_pnl"]/1000, w,
       color="C2", alpha=0.8,
       label=f"+ long 100$\\times$ {K_hedge}-put hedge")
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels(cov["label"], fontsize=8)
ax.set_ylabel("PnL ($ thousands)")
ax.set_title(f"Stress replay: short 100$\\times$ {K_short}-strike SPY 30-DTE put as of "
             f"{today_date.date()} (F={F_today:.0f}), repriced under "
             f"{len(cov)} historical $\\Delta F\\leq-3\\%$ event smiles")
ax.legend(loc="lower right")

for i in range(len(cov)):
    cov_pct = cov["hedge_coverage_pct"].iloc[i]
    if pd.notna(cov_pct):
        y_top = max(cov["hedged_pnl"].iloc[i],
                    cov["unhedged_pnl"].iloc[i]) / 1000
        ax.annotate(f"cov: {cov_pct:.0f}%",
                    xy=(i, y_top),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=8, color="C2", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig25_portfolio_stress_replay.png"))
plt.close()
print(f"  -> {OUT_DIR}/fig25_portfolio_stress_replay.png")


# ============================================================
# Summary numbers for paper text
# ============================================================
print("\n\n=== Numbers for paper Appendix ===")
print(f"# events (drop <= -3%, 2014-2023):  {len(events)}")
print(f"Worst single-day drop:  {events.iloc[0]['return_pct']:.2f}% on "
      f"{events.index[0].date()}")
print(f"\nEvent-window trajectory (top 8 events, +/-5 days):")
print(agg.round(4).to_string(index=False))
print(f"\nPortfolio: short {abs(position[0]['qty'])}x put @ {K_short}, "
      f"hedge: long {abs(hedge[0]['qty'])}x put @ {K_hedge}")
print(f"Today: F={F_today:.1f}, T=30/365, r={r_today*100:.3f}%")
print(cov[["label", "return_pct", "unhedged_pnl", "hedged_pnl",
           "hedge_coverage_pct"]].round(2).to_string(index=False))
print(f"\nMedian hedge coverage across events:  "
      f"{cov['hedge_coverage_pct'].median():.1f}%")
print(f"Worst unhedged loss:  ${cov['unhedged_pnl'].min():,.0f}")
print(f"Worst hedged loss:    ${cov['hedged_pnl'].min():,.0f}")
