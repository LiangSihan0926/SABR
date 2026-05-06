"""Interactive / animated visualisation helpers for the SABR project.

Three deliverables:

(A) ``make_smile_animation_gif(...)`` produces a 4-panel matplotlib
    animation (one panel per SABR parameter) sweeping that parameter
    over its range while the others are held fixed. Saved as a GIF
    so it can be embedded directly in the README and auto-played by
    GitHub.

(B) ``make_interactive_smile_html(...)`` produces a self-contained
    HTML page with four Plotly-based animations stacked vertically,
    each with its own slider for one parameter. Loads in any browser
    via the Plotly CDN; suitable for hosting on GitHub Pages.

(C) ``make_interactive_timeseries_html(...)`` produces a self-contained
    HTML with a 3-row Plotly time series of the calibrated
    (alpha, rho, nu) over 2014-2023, with stress events marked as
    red vertical lines. Hover shows the date and exact parameter
    values; the legend can be clicked to toggle each trace.

The HTML files are deliberately written with the Plotly script loaded
from a CDN (``include_plotlyjs="cdn"``) so each file weighs a few KB
rather than a few MB.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .sabr import sabr_vol


# ============================================================
# (A) Animated GIF: 4-panel param sweep
# ============================================================
def make_smile_animation_gif(
    out_path: str,
    F: float = 100.0,
    T: float = 1.0,
    base: tuple = (0.20, 0.5, -0.30, 0.40),
    n_frames: int = 30,
    fps: int = 10,
    K_grid: np.ndarray | None = None,
):
    """Produce a 4-panel animated GIF showing how the smile changes
    as one SABR parameter sweeps over its typical range.

    Each panel sweeps one of (alpha, beta, rho, nu); the other three
    stay at ``base`` values. Loops smoothly using a sine-wave time
    parameter so playback feels continuous.
    """
    if K_grid is None:
        K_grid = np.linspace(70, 130, 121)

    alpha0, beta0, rho0, nu0 = base

    # Sweep ranges per parameter
    alpha_lo, alpha_hi = 0.10, 0.30
    rho_lo,   rho_hi   = -0.7, +0.4
    nu_lo,    nu_hi    = 0.10, 0.80
    beta_grid = [0.0, 0.5, 1.0]      # discrete

    def _wave(i):
        # 0 .. 1 .. 0 over the full loop, smooth
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * i / n_frames)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    axes = axes.flatten()

    base_vol = sabr_vol(K_grid, F, T, alpha0, beta0, rho0, nu0) * 100
    lines = []
    titles = [
        rf"Sweep $\alpha$ (level)",
        rf"Sweep $\beta$ (backbone)",
        rf"Sweep $\rho$ (skew)",
        rf"Sweep $\nu$ (curvature)",
    ]
    for ax, title in zip(axes, titles):
        line, = ax.plot(K_grid, base_vol, lw=2, color="C3")
        ax.axvline(F, color="grey", lw=0.6, ls="--")
        ax.set_xlim(K_grid.min(), K_grid.max())
        ax.set_ylim(0.0, 6.5)
        ax.set_xlabel("K")
        ax.set_ylabel("σ (%)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        lines.append(line)

    text_handles = [
        ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        for ax in axes
    ]

    def _update(i):
        t = _wave(i)
        a = alpha_lo + (alpha_hi - alpha_lo) * t
        r = rho_lo   + (rho_hi   - rho_lo)   * t
        nu = nu_lo   + (nu_hi    - nu_lo)    * t
        # beta cycles through discrete values
        bidx = int(t * (len(beta_grid) - 0.001))
        b = beta_grid[bidx]

        sweeps = [
            (a,     beta0, rho0, nu0),
            (alpha0, b,    rho0, nu0),
            (alpha0, beta0, r,   nu0),
            (alpha0, beta0, rho0, nu),
        ]
        labels = [
            f"α={a:.3f}",
            f"β={b:.1f}",
            f"ρ={r:+.2f}",
            f"ν={nu:.2f}",
        ]
        for line, txt, params, lab in zip(lines, text_handles, sweeps, labels):
            v = sabr_vol(K_grid, F, T, *params) * 100
            line.set_ydata(v)
            txt.set_text(lab)
        return lines + text_handles

    fig.suptitle("SABR smile — interactive parameter sweep "
                 f"(F={F}, T={T})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


# ============================================================
# (B) Interactive Plotly HTML: 4-panel slider explorer
# ============================================================
def make_interactive_smile_html(
    out_path: str,
    F: float = 100.0,
    T: float = 1.0,
    base: tuple = (0.20, 0.5, -0.30, 0.40),
    n_steps: int = 21,
    K_grid: np.ndarray | None = None,
):
    """Build a single HTML page containing four stacked Plotly figures.
    Each figure has its own slider that animates one SABR parameter
    while the other three stay fixed at ``base``.

    Output is a standalone HTML page suitable for hosting on GitHub
    Pages or opening locally in a browser.
    """
    if K_grid is None:
        K_grid = np.linspace(70, 130, 121)

    alpha0, beta0, rho0, nu0 = base

    sweep_specs = [
        ("alpha", "α", np.linspace(0.10, 0.30, n_steps),
         lambda v: (v, beta0, rho0, nu0)),
        ("beta",  "β", np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
         lambda v: (alpha0, v, rho0, nu0)),
        ("rho",   "ρ", np.linspace(-0.7, +0.4, n_steps),
         lambda v: (alpha0, beta0, v, nu0)),
        ("nu",    "ν", np.linspace(0.10, 0.80, n_steps),
         lambda v: (alpha0, beta0, rho0, v)),
    ]

    div_blobs = []
    for slug, glyph, vgrid, params_of in sweep_specs:
        # pre-compute frames
        frames = []
        for v in vgrid:
            params = params_of(float(v))
            sigma = sabr_vol(K_grid, F, T, *params) * 100.0
            frames.append(go.Frame(
                name=f"{v:.3f}",
                data=[go.Scatter(x=K_grid, y=sigma,
                                 mode="lines", line=dict(color="#B31B1B", width=3))],
            ))

        init_sigma = sabr_vol(K_grid, F, T, *params_of(float(vgrid[0]))) * 100.0
        fig = go.Figure(
            data=[go.Scatter(x=K_grid, y=init_sigma,
                             mode="lines", line=dict(color="#B31B1B", width=3),
                             name="SABR smile")],
            frames=frames,
        )
        fig.update_layout(
            title=dict(
                text=f"SABR smile — sweep <b>{glyph}</b>"
                     f"  (F={F}, T={T}; "
                     f"others fixed: α={alpha0}, β={beta0}, ρ={rho0}, ν={nu0})",
                x=0.02, font=dict(size=13),
            ),
            xaxis=dict(title="Strike K"),
            yaxis=dict(title="Implied volatility (%)", range=[0.0, 7.0]),
            height=380, margin=dict(l=50, r=20, t=60, b=80),
            sliders=[{
                "active": 0,
                "currentvalue": {"prefix": f"{glyph} = ", "font": {"size": 13}},
                "pad": {"b": 10, "t": 30},
                "len": 0.85,
                "x": 0.08,
                "steps": [
                    {"args": [[f.name], {"frame": {"duration": 0},
                                          "mode": "immediate"}],
                     "label": f.name, "method": "animate"}
                    for f in frames
                ],
            }],
            updatemenus=[{
                "type": "buttons", "showactive": False,
                "x": 0.02, "y": -0.13,
                "buttons": [
                    {"args": [None, {"frame": {"duration": 200, "redraw": True},
                                      "fromcurrent": True}],
                     "label": "▶ Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}],
                     "label": "❚❚ Pause", "method": "animate"},
                ],
            }],
        )
        div_blobs.append(fig.to_html(full_html=False,
                                     include_plotlyjs="cdn" if slug == "alpha" else False,
                                     div_id=f"plot_{slug}"))

    page = _wrap_html(
        title="SABR Smile Explorer",
        intro=("<p>Drag each slider (or hit ▶) to see how a single SABR "
               "parameter reshapes the implied-volatility smile, with the "
               "others held fixed at canonical values. "
               "Generated by "
               "<code>src/interactive_viz.py:make_interactive_smile_html</code>."
               "</p>"),
        bodies=div_blobs,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(page)
    return out_path


# ============================================================
# (C) Interactive Plotly HTML: calibration time-series with
#     stress-event markers
# ============================================================
def make_interactive_timeseries_html(
    out_path: str,
    cal_df: pd.DataFrame,
    event_dates: Iterable,
    ticker: str = "SPY",
    dte: int = 30,
    beta: float = 0.5,
):
    """Build a 3-row Plotly time-series HTML for the calibrated
    (alpha, rho, nu) panel, with vertical lines at every stress event.

    cal_df should be the cache/calibration_results.parquet DataFrame.
    event_dates is an iterable of pandas-parseable dates.
    """
    sub = cal_df.query("ticker == @ticker and dte == @dte and beta == @beta").copy()
    sub["trade_date"] = pd.to_datetime(sub["trade_date"])
    sub = sub.sort_values("trade_date")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("α (volatility level)",
                        "ρ (skew)",
                        "ν (curvature)"),
    )
    common = dict(mode="lines+markers", marker=dict(size=5),
                  line=dict(width=1.6))

    fig.add_trace(go.Scatter(x=sub["trade_date"], y=sub["alpha"],
                             name="α", line=dict(color="#1f77b4", width=1.6),
                             marker=dict(size=5),
                             hovertemplate="%{x|%Y-%m-%d}<br>α = %{y:.3f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sub["trade_date"], y=sub["rho"],
                             name="ρ", line=dict(color="#2ca02c", width=1.6),
                             marker=dict(size=5),
                             hovertemplate="%{x|%Y-%m-%d}<br>ρ = %{y:+.3f}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=sub["trade_date"], y=sub["nu"],
                             name="ν", line=dict(color="#d62728", width=1.6),
                             marker=dict(size=5),
                             hovertemplate="%{x|%Y-%m-%d}<br>ν = %{y:.3f}<extra></extra>"),
                  row=3, col=1)

    # Stress event vertical lines
    for d in pd.DatetimeIndex(event_dates):
        for r in (1, 2, 3):
            fig.add_vline(x=d, line=dict(color="rgba(220,40,40,0.35)",
                                          width=1, dash="dot"),
                          row=r, col=1)

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_xaxes(title_text="Trade date", row=3, col=1)
    fig.update_layout(
        title=dict(text=(f"<b>{ticker} {dte}-DTE</b> — calibrated SABR "
                          f"parameter time series  (β={beta})  with stress events "
                          "(dashed red lines, ΔF ≤ −3%)"),
                   x=0.02, font=dict(size=13)),
        height=720,
        hovermode="x unified",
        legend=dict(orientation="h", x=0.02, y=1.05),
        margin=dict(l=60, r=30, t=80, b=50),
    )

    page = _wrap_html(
        title=f"{ticker} {dte}-DTE SABR Calibration Time Series",
        intro=("<p>Hover any trace to see the calibrated parameter on that "
               "trade date; click legend entries to hide/show; drag a "
               "rectangle on any panel to zoom (double-click to reset). "
               "Dashed red verticals mark single-day SPY drops of 3% or more "
               "(2014–2023, 32 events).</p>"),
        bodies=[fig.to_html(full_html=False, include_plotlyjs="cdn",
                            div_id="ts_plot")],
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(page)
    return out_path


# ============================================================
# (D) Interactive stress-replay explorer
# ============================================================
def make_interactive_stress_html(
    out_path: str,
    events_df: pd.DataFrame,
    baseline_state,
    today_F: float,
    today_T: float,
    today_r: float,
    spy_options: pd.DataFrame,
    rates_df: pd.DataFrame,
    short_strike_pct: float = 0.95,
    n_contracts: int = 100,
    hedge_strike_grid: tuple = (0.85, 0.88, 0.90, 0.92, 0.95),
    beta: float = 0.5,
):
    """Build a single-page Plotly explorer for the event-anchored
    stress replay. The user picks the hedge strike via a slider and
    sees:

      - one bar per historical event showing unhedged vs hedged PnL
      - a callout text in the figure title with the resulting 95% VaR,
        CVaR, worst-case loss, and median hedge coverage

    Inputs
    ------
    events_df       : output of find_stress_events
    baseline_state  : MarketState (today's calibrated SABR + market)
    today_F/T/r     : market state today
    spy_options     : SPY options DataFrame (cache/spy_options_filtered)
    rates_df        : FRED yield DataFrame (forward-filled)
    short_strike_pct: e.g. 0.95 -> short the 95%-strike put
    hedge_strike_grid: tuple of hedge strike % values to enumerate

    The function calibrates SABR on each event date once, then
    re-prices the (short put + long hedge) portfolio for each
    (event, hedge_strike_pct) combination. Output is one standalone
    HTML page.
    """
    from .data_loader import build_smile
    from .calibration import calibrate_smile_panel
    from .event_stress import (
        MarketState, replay_scenarios, hedge_coverage,
        var_cvar, hedge_cost,
    )

    # 1. Build per-event MarketState list (calibrating SABR on event days)
    print("  Calibrating SABR on each event for stress explorer ...")
    stresses = []
    event_labels = []
    event_returns = []
    for ev_date, ev_row in events_df.iterrows():
        try:
            sm = build_smile(spy_options, rates_df,
                             ev_date.strftime("%Y-%m-%d"), dte=30)
            fit = calibrate_smile_panel(sm, beta=beta)
        except Exception:
            continue
        shock = ev_row["return_pct"] / 100.0
        stresses.append(MarketState(
            F=today_F * (1 + shock), T=today_T, r=today_r,
            alpha=fit.alpha, beta=beta, rho=fit.rho, nu=fit.nu,
            label=str(ev_date.date()),
        ))
        event_labels.append(str(ev_date.date()))
        event_returns.append(shock * 100.0)

    K_short = round(today_F * short_strike_pct)
    position = [{"strike": K_short, "type": "P", "qty": -n_contracts}]

    # Position-only baseline value once (constant across hedge variants)
    pos_only_pnl = replay_scenarios(position, baseline_state, stresses)
    unhedged_pnl = pos_only_pnl["pnl"].to_numpy()

    # 2. Compute hedged variants per hedge strike
    n_evt = len(stresses)
    n_hedge = len(hedge_strike_grid) + 1   # +1 for "no hedge"
    hedge_labels = ["no hedge"] + [f"long {n_contracts}x {int(today_F*p)}-put"
                                    for p in hedge_strike_grid]

    pnl_matrix    = np.zeros((n_hedge, n_evt))
    cost_per_var  = np.zeros(n_hedge)
    cov_per_var   = np.zeros(n_hedge)
    var_per_var   = np.zeros(n_hedge)
    cvar_per_var  = np.zeros(n_hedge)
    worst_per_var = np.zeros(n_hedge)

    for j, p in enumerate([None, *hedge_strike_grid]):
        if p is None:
            hedge = []
        else:
            K_h = round(today_F * p)
            hedge = [{"strike": K_h, "type": "P", "qty": +n_contracts}]
        if hedge:
            res = hedge_coverage(position, hedge, baseline_state, stresses)
            pnl_matrix[j] = res["hedged_pnl"].to_numpy()
            cost_per_var[j] = hedge_cost(hedge, baseline_state)
            cov_per_var[j] = float(np.nanmedian(res["hedge_coverage_pct"]))
        else:
            pnl_matrix[j] = unhedged_pnl
            cost_per_var[j] = 0.0
            cov_per_var[j] = 0.0
        v, c = var_cvar(pnl_matrix[j], confidence=0.95)
        var_per_var[j]   = v
        cvar_per_var[j]  = c
        worst_per_var[j] = -pnl_matrix[j].min()

    # 3. Build Plotly figure with one frame per hedge variant
    # Sort events by return (worst on the left)
    order = np.argsort(event_returns)
    event_labels_sorted  = [event_labels[i]  for i in order]
    event_returns_sorted = [event_returns[i] for i in order]

    def _bars(idx):
        return go.Bar(
            x=event_labels_sorted,
            y=pnl_matrix[idx][order] / 1000.0,
            marker_color=["#B31B1B" if idx == 0 else "#2E8B3D"
                          for _ in event_labels_sorted],
            hovertemplate=("<b>%{x}</b><br>"
                            "PnL: $%{y:.2f}k<extra></extra>"),
            name=hedge_labels[idx],
        )

    def _annotation(idx):
        return (f"<b>{hedge_labels[idx]}</b><br>"
                f"Hedge premium: ${cost_per_var[idx]:.2f}<br>"
                f"95% VaR:  ${var_per_var[idx]:,.0f}<br>"
                f"95% CVaR: ${cvar_per_var[idx]:,.0f}<br>"
                f"Worst case: ${worst_per_var[idx]:,.0f}<br>"
                f"Median coverage: {cov_per_var[idx]:.1f}%")

    init_idx = 0
    frames = [
        go.Frame(
            name=str(j),
            data=[_bars(j)],
            layout=go.Layout(annotations=[dict(
                text=_annotation(j),
                xref="paper", yref="paper",
                x=1.0, y=1.0, xanchor="right", yanchor="top",
                showarrow=False, align="right",
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="#888", borderwidth=1, borderpad=8,
                font=dict(size=12, family="monospace"),
            )]),
        )
        for j in range(n_hedge)
    ]

    fig = go.Figure(
        data=[_bars(init_idx)],
        frames=frames,
    )
    fig.update_layout(
        title=dict(text=(f"<b>Stress replay explorer</b> &nbsp; "
                          f"short {n_contracts}× {K_short}-put on SPY "
                          f"30-DTE  (today F = {today_F:.0f}). "
                          "Drag the slider to change the long-put hedge."),
                   x=0.02, font=dict(size=13)),
        xaxis=dict(title="Historical event (worst → least)", tickangle=-30),
        yaxis=dict(title="Per-event PnL ($ thousands; negative = loss)"),
        height=560, margin=dict(l=60, r=240, t=80, b=120),
        annotations=[dict(
            text=_annotation(init_idx),
            xref="paper", yref="paper",
            x=1.0, y=1.0, xanchor="right", yanchor="top",
            showarrow=False, align="right",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#888", borderwidth=1, borderpad=8,
            font=dict(size=12, family="monospace"),
        )],
        showlegend=False,
        sliders=[{
            "active": init_idx,
            "currentvalue": {"prefix": "Hedge: ",
                              "font": {"size": 13}},
            "pad": {"b": 10, "t": 30},
            "len": 0.85,
            "x": 0.08,
            "steps": [
                {"args": [[str(j)], {"frame": {"duration": 0},
                                      "mode": "immediate"}],
                 "label": hedge_labels[j], "method": "animate"}
                for j in range(n_hedge)
            ],
        }],
    )

    page = _wrap_html(
        title="SABR Stress Replay Explorer",
        intro=(f"<p>This page replays each of the {n_evt} historical SPY "
               f"single-day drops of −3% or worse (2014–2023) onto a "
               f"sample portfolio held <b>today</b> "
               f"(F = {today_F:.0f}, T = 30 days). The position is a "
               f"<b>short {n_contracts}× {K_short}-strike put</b>. "
               "Drag the slider below to swap in different long-put "
               "hedges (or no hedge); the per-event PnL bars and the "
               "tail-risk readout in the upper-right update live. "
               "Built by <code>src/interactive_viz.py</code>.</p>"),
        bodies=[fig.to_html(full_html=False, include_plotlyjs="cdn",
                            div_id="stress_plot")],
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(page)
    return out_path


# ============================================================
# Internal: tiny HTML wrapper
# ============================================================
def _wrap_html(title: str, intro: str, bodies: list) -> str:
    """Concatenate Plotly div blobs into one styled standalone page."""
    body = "\n<hr/>\n".join(bodies)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                       Helvetica, Arial, sans-serif;
          max-width: 1100px; margin: 24px auto; padding: 0 16px;
          color: #222; }}
  h1 {{ color: #B31B1B; margin-bottom: 4px; }}
  .subtitle {{ color: #555; font-size: 14px; margin-bottom: 16px; }}
  hr {{ border: 0; border-top: 1px solid #eee; margin: 24px 0; }}
  code, kbd {{ background: #f6f6f6; padding: 1px 6px; border-radius: 3px;
                font-size: 0.92em; }}
  a {{ color: #B31B1B; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">SABR final project — interactive demo.
Source: <a href="https://github.com/LiangSihan0926/SABR">github.com/LiangSihan0926/SABR</a>.</div>
{intro}
{body}
</body>
</html>
"""
