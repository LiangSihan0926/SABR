"""Build the interactive-visualisation deliverables.

Outputs:
    figures/sabr_animation.gif                animated GIF for README
    docs/interactive_smile.html               4-slider Plotly smile explorer
    docs/interactive_timeseries.html          hoverable parameter time series

Run from the project root:
    python3 notebooks/_build_interactive.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
CACHE_DIR = os.path.join(ROOT, "cache")
DATA_DIR  = os.path.join(ROOT, "Data")

from src.interactive_viz import (
    make_smile_animation_gif,
    make_interactive_smile_html,
    make_interactive_timeseries_html,
    make_interactive_stress_html,
)
from src.event_stress import find_stress_events, MarketState
from src.data_loader import load_fred_yields, build_smile
from src.calibration import calibrate_smile_panel


# ---------------------------------------------------------------
# (A) Animated GIF
# ---------------------------------------------------------------
print("Building sabr_animation.gif ...")
gif_path = os.path.join(ROOT, "figures", "sabr_animation.gif")
make_smile_animation_gif(gif_path, n_frames=30, fps=10)
print(f"  -> {gif_path}  ({os.path.getsize(gif_path)/1024:.0f} KB)")


# ---------------------------------------------------------------
# (B) Interactive smile explorer (4 sliders)
# ---------------------------------------------------------------
print("\nBuilding docs/interactive_smile.html ...")
html_path = os.path.join(ROOT, "docs", "interactive_smile.html")
make_interactive_smile_html(html_path, n_steps=21)
print(f"  -> {html_path}  ({os.path.getsize(html_path)/1024:.0f} KB)")


# ---------------------------------------------------------------
# (C) Interactive parameter time series with stress events
# ---------------------------------------------------------------
print("\nBuilding docs/interactive_timeseries.html ...")
spy = pd.read_parquet(os.path.join(CACHE_DIR, "spy_options_filtered.parquet"))
cal = pd.read_parquet(os.path.join(CACHE_DIR, "calibration_results.parquet"))
events = find_stress_events(spy, return_threshold=-0.03)

ts_path = os.path.join(ROOT, "docs", "interactive_timeseries.html")
make_interactive_timeseries_html(ts_path, cal, events.index,
                                  ticker="SPY", dte=30, beta=0.5)
print(f"  -> {ts_path}  ({os.path.getsize(ts_path)/1024:.0f} KB)")

# ---------------------------------------------------------------
# (D) Interactive stress-replay explorer
# ---------------------------------------------------------------
print("\nBuilding docs/interactive_stress.html ...")
rates = load_fred_yields(DATA_DIR).ffill()
today_date  = spy["QUOTE_DATE"].max()
today_smile = build_smile(spy, rates, today_date.strftime("%Y-%m-%d"), dte=30)
today_fit   = calibrate_smile_panel(today_smile, beta=0.5)
F_today, T_today, r_today = (today_smile.attrs["F"],
                              today_smile.attrs["expire_dte"] / 365.25,
                              today_smile.attrs["r"])
baseline = MarketState(
    F=F_today, T=T_today, r=r_today,
    alpha=today_fit.alpha, beta=0.5,
    rho=today_fit.rho, nu=today_fit.nu,
    label=f"today ({today_date.date()})",
)
stress_path = os.path.join(ROOT, "docs", "interactive_stress.html")
make_interactive_stress_html(
    stress_path, events, baseline,
    today_F=F_today, today_T=T_today, today_r=r_today,
    spy_options=spy, rates_df=rates,
)
print(f"  -> {stress_path}  ({os.path.getsize(stress_path)/1024:.0f} KB)")


print("\nAll four interactive deliverables built.")
print("To preview locally:")
print(f"  open {html_path}")
print(f"  open {ts_path}")
print(f"  open {stress_path}")
