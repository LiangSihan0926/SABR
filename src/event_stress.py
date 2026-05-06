"""Event-anchored regime analysis and portfolio scenario replay.

This module is intentionally distinct from the alpha-bucket regime view
in :mod:`src.regime` (which classifies every trade day by SABR alpha
level into Low/Mid1/Mid2/High) and from the single-position hedge
decomposition in :mod:`src.stress_test` (which decomposes PnL into
delta/vega legs at one strike).

Two complementary deliverables here:

(A) **Event-anchored regime tagging.** Identify historical *stress events*
    --- single-day drops of the underlying that exceed a return
    threshold (default -3%). Tag every trade date by its proximity to
    such events: ``event``, ``pre``, ``post``, or ``normal``. Aggregate
    calibrated SABR parameters across these tags to expose how the smile
    behaves around (rather than during) crises.

(B) **Portfolio scenario replay.** Given today's calibrated SABR state
    and a multi-leg vanilla-options portfolio, reprice the portfolio
    under each historical event's calibrated SABR state. Report the PnL
    distribution across events, the worst-case scenario, and the
    coverage achieved by an overlay hedge.

The two together answer a different question from the alpha-bucket /
single-position approach: "If today's portfolio went through one of
the historical shock days, how badly would it perform on aggregate, and
how much of that loss does my candidate hedge actually cover?"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sabr import sabr_vol
from .black import black_call, black_put


# ============================================================
# A. Event-anchored regime
# ============================================================
def find_stress_events(options_df, return_threshold=-0.03):
    """Find trade dates with single-day spot drop <= ``return_threshold``.

    Parameters
    ----------
    options_df : DataFrame
        Must contain ``QUOTE_DATE`` (datetime) and ``UNDERLYING_LAST``
        (numeric) columns. The cached parquet from Module 3 satisfies
        this.
    return_threshold : float
        Negative number; default -0.03 picks days that fell at least 3%.

    Returns
    -------
    DataFrame indexed by ``event_date`` with columns
        ``F_prev``, ``F_event``, ``return_pct`` (in percent).
    Sorted from worst (most negative) to least.
    """
    spot = (options_df.groupby("QUOTE_DATE")["UNDERLYING_LAST"]
                       .mean().sort_index())
    ret = spot.pct_change()
    mask = ret <= return_threshold
    out = pd.DataFrame({
        "F_prev":     spot.shift(1).loc[mask],
        "F_event":    spot.loc[mask],
        "return_pct": ret.loc[mask] * 100.0,
    })
    out.index.name = "event_date"
    return out.sort_values("return_pct")


def label_event_regime(trade_dates, event_dates, window_days=5):
    """Return a Series tagging each trade date as one of
    ``{'event', 'pre', 'post', 'normal'}``.

    A date is ``'event'`` if it is itself an event; ``'pre'`` if within
    ``window_days`` calendar days *before* an event; ``'post'`` if
    within ``window_days`` *after*; ``'normal'`` otherwise. ``'event'``
    takes precedence; the first matching window otherwise wins.
    """
    trade_dates = pd.DatetimeIndex(trade_dates).normalize()
    event_dates = pd.DatetimeIndex(event_dates).normalize()
    event_set = set(event_dates)

    labels = pd.Series("normal", index=trade_dates, dtype=object)
    for d in trade_dates:
        if d in event_set:
            labels[d] = "event"
            continue
        for ev in event_dates:
            delta_days = (ev - d).days
            if 0 < delta_days <= window_days:
                if labels[d] == "normal":
                    labels[d] = "pre"
                break
            if 0 < -delta_days <= window_days:
                if labels[d] == "normal":
                    labels[d] = "post"
                break
    return labels


# Note: a previously-defined `event_regime_summary()` aggregation has
# been intentionally omitted. The reference work this project compares
# against reports a per-bucket median table including a "rho-at-boundary
# %" column; we deliberately do not reproduce that table here. The
# event-anchored regime tagging in `label_event_regime` is retained as
# an internal utility for selecting calibration dates that feed into the
# portfolio scenario replay below; the regime tags are not themselves
# the reported deliverable.


# ============================================================
# B. Portfolio scenario replay
# ============================================================
@dataclass
class MarketState:
    """Bundles everything needed to price a vanilla portfolio under SABR."""
    F:     float
    T:     float
    r:     float
    alpha: float
    beta:  float
    rho:   float
    nu:    float
    label: str = ""        # free-form description for reporting


def portfolio_value(position, state: MarketState) -> float:
    """Mark-to-market value of a multi-leg vanilla portfolio.

    ``position`` is a list of dicts; each dict has keys
        ``strike`` (float), ``type`` ('C' or 'P'), ``qty`` (signed).
    A negative qty represents a short leg.
    """
    if not position:
        return 0.0
    Ks = np.array([opt["strike"] for opt in position], dtype=float)
    sigmas = np.atleast_1d(
        sabr_vol(Ks, state.F, state.T, state.alpha,
                 state.beta, state.rho, state.nu)
    )
    total = 0.0
    for opt, sigma in zip(position, sigmas):
        K = float(opt["strike"])
        sigma = float(sigma)
        if opt["type"].upper().startswith("C"):
            price = black_call(state.F, K, sigma, state.T, state.r)
        else:
            price = black_put(state.F, K, sigma, state.T, state.r)
        total += float(opt["qty"]) * float(price)
    return total


def replay_scenarios(position, baseline: MarketState, stresses):
    """Reprice ``position`` under each ``MarketState`` in ``stresses``
    and return a per-scenario PnL DataFrame.

    Returns columns
        label, F_stress, return_pct, base_value, stress_value,
        pnl, pnl_pct.
    """
    base_value = portfolio_value(position, baseline)
    rows = []
    for s in stresses:
        sv = portfolio_value(position, s)
        rows.append({
            "label":        s.label,
            "F_stress":     s.F,
            "return_pct":   100.0 * (s.F / baseline.F - 1.0),
            "base_value":   base_value,
            "stress_value": sv,
            "pnl":          sv - base_value,
            "pnl_pct":      (sv - base_value) /
                            max(abs(base_value), 1e-12) * 100.0,
        })
    return pd.DataFrame(rows)


def hedge_coverage(position, hedge, baseline: MarketState, stresses):
    """Compare unhedged vs hedged scenario PnL.

    Returns DataFrame with columns
        label, return_pct, unhedged_pnl, hedged_pnl, hedge_coverage_pct.

    ``hedge_coverage_pct`` is ``100 * (1 - |hedged|/|unhedged|)``,
    so 100% means the hedge fully neutralises the loss; 0% means it
    didn't help; negative means the hedge made things worse.
    """
    unhedged = replay_scenarios(position, baseline, stresses)
    combined = list(position) + list(hedge)
    hedged   = replay_scenarios(combined, baseline, stresses)

    out = unhedged[["label", "return_pct"]].copy()
    out["unhedged_pnl"] = unhedged["pnl"]
    out["hedged_pnl"]   = hedged["pnl"]
    denom = unhedged["pnl"].abs().replace(0.0, np.nan)
    out["hedge_coverage_pct"] = 100.0 * (1.0 - hedged["pnl"].abs() / denom)
    return out
