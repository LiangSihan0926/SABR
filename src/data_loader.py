"""Data loaders for the SABR final project.

Two data sources:
    * FRED daily Treasury CMT yields (DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2)
      → risk-free rate for Black's formula.
    * optionsdx.com end-of-day options chain (.txt, comma+space separated)
      → market smile  σ_mkt(K, T) for each (ticker, trade_date, expiry).

High-level workflow for Module 4 (calibration):
    rates = load_fred_yields(data_dir)
    opts  = load_options(underlying='spy', data_dir=data_dir, years=[2022])
    smile = build_smile(opts, rates, trade_date='2022-01-03', dte=30)
"""
from __future__ import annotations

import os
import glob
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------
# FRED yields
# ---------------------------------------------------------------
FRED_TENORS = {
    "DGS1MO": 1.0 / 12.0,
    "DGS3MO": 3.0 / 12.0,
    "DGS6MO": 6.0 / 12.0,
    "DGS1":   1.0,
    "DGS2":   2.0,
}


def load_fred_yields(data_dir):
    """Load the five FRED daily Treasury yield series into one DataFrame.

    Returns
    -------
    DataFrame indexed by observation_date (pandas Timestamp), with columns
    `DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2` holding the yield in decimal form
    (e.g. 0.0525 for 5.25%).  Missing observations (holidays etc.) are NaN
    after the merge; the caller can forward-fill as needed.
    """
    frames = []
    for name in FRED_TENORS:
        path = os.path.join(data_dir, f"{name}.csv")
        s = pd.read_csv(path, parse_dates=["observation_date"])
        s = s.rename(columns={"observation_date": "date"})
        s[name] = pd.to_numeric(s[name], errors="coerce") / 100.0
        frames.append(s.set_index("date")[name])
    df = pd.concat(frames, axis=1).sort_index()
    return df


def interpolate_yield(rates_df, date, tenor_years):
    """Linear interpolation on the CMT curve for (date, tenor_years).

    * If `date` is not in the index (holiday), the most recent prior row
      is used.
    * For tenors shorter than DGS1MO (~1/12 yr) the DGS1MO yield is used.
    * For tenors longer than DGS2 (2 yr) the DGS2 yield is used.
    """
    date = pd.Timestamp(date)
    idx = rates_df.index
    if date not in idx:
        idx_before = idx[idx <= date]
        if len(idx_before) == 0:
            raise ValueError(f"no FRED yield available on or before {date}")
        date = idx_before[-1]
    row = rates_df.loc[date].dropna()
    if row.empty:
        # fall back one business day at a time
        pos = idx.get_loc(date)
        while pos > 0 and row.empty:
            pos -= 1
            row = rates_df.iloc[pos].dropna()
        if row.empty:
            raise ValueError(f"no FRED yield near {date}")
    tenors = np.array([FRED_TENORS[c] for c in row.index])
    yields = row.values.astype(float)
    order = np.argsort(tenors)
    tenors, yields = tenors[order], yields[order]
    tau = float(tenor_years)
    if tau <= tenors[0]:
        return float(yields[0])
    if tau >= tenors[-1]:
        return float(yields[-1])
    return float(np.interp(tau, tenors, yields))


# ---------------------------------------------------------------
# optionsdx options chains
# ---------------------------------------------------------------
# Columns we keep after normalization; the raw file has ~33 columns.
_RAW_COLS = [
    "QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE",
    "C_IV", "C_BID", "C_ASK", "C_VOLUME",
    "P_IV", "P_BID", "P_ASK", "P_VOLUME",
    "STRIKE", "STRIKE_DISTANCE_PCT",
]


def _normalize_columns(df):
    """optionsdx headers look like `[QUOTE_DATE]` — strip the brackets
    and whitespace, upper-case them."""
    df.columns = [c.strip().strip("[]").strip().upper() for c in df.columns]
    return df


def load_options_file(path, keep_cols=_RAW_COLS, max_moneyness=0.30,
                      min_dte=5, max_dte=365):
    """Load one optionsdx EOD monthly file, return a cleaned DataFrame.

    Parameters
    ----------
    path : str
    keep_cols : list[str]
        Columns to retain post-normalization.
    max_moneyness : float
        Drop rows with |STRIKE_DISTANCE_PCT| > this (keeps near-the-money
        part of the smile; default 0.30 ≈ ±30% of spot).
    min_dte, max_dte : int
        Drop rows with DTE outside this window.  The 0-DTE row at top of
        each file (same-day expiry) is dropped as it is not useful for
        calibration.
    """
    df = pd.read_csv(path, sep=",", skipinitialspace=True)
    df = _normalize_columns(df)
    df = df[keep_cols].copy()

    # parse types
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"], errors="coerce")
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"], errors="coerce")
    for c in ["UNDERLYING_LAST", "DTE", "C_IV", "C_BID", "C_ASK",
              "C_VOLUME", "P_IV", "P_BID", "P_ASK", "P_VOLUME",
              "STRIKE", "STRIKE_DISTANCE_PCT"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # basic cleaning
    df = df.dropna(subset=["QUOTE_DATE", "EXPIRE_DATE", "STRIKE",
                            "UNDERLYING_LAST"])
    df = df[(df["DTE"] >= min_dte) & (df["DTE"] <= max_dte)]
    df = df[df["STRIKE_DISTANCE_PCT"].abs() <= max_moneyness]
    df = df[df["STRIKE"] > 0]
    return df.reset_index(drop=True)


def load_options(underlying, data_dir, years=None, **kwargs):
    """Load and concatenate many optionsdx monthly files.

    Parameters
    ----------
    underlying : 'spy' | 'qqq' | ... (lower-case folder name under data_dir)
    data_dir   : root Data/ folder
    years      : iterable of ints; None = all years found on disk.
    kwargs     : forwarded to :func:`load_options_file`
    """
    folder = os.path.join(data_dir, underlying)
    paths = sorted(glob.glob(os.path.join(folder, f"{underlying}_eod_*.txt")))
    if years is not None:
        years = set(int(y) for y in years)
        paths = [p for p in paths if int(os.path.basename(p)[-10:-6]) in years]
    if not paths:
        raise FileNotFoundError(f"no files found in {folder} for years={years}")
    frames = [load_options_file(p, **kwargs) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["TICKER"] = underlying.upper()
    return df


# ---------------------------------------------------------------
# Smile construction for a single (ticker, trade_date, expiry)
# ---------------------------------------------------------------
def extract_forward(smile_df, r):
    """Implied forward from put-call parity at the strike minimizing
    |C_mid - P_mid|.

    At an ATM-equivalent strike K*,   C - P = D(F - K)  ⇒  F = K* + (C - P)*exp(r T).
    Selecting K* by minimum |C_mid - P_mid| (rather than by proximity to
    spot) puts K* at the strike where call and put quotes are most
    consistent under European parity, which reduces noise from the
    bid-ask spread at ITM strikes.
    """
    df = smile_df.copy()
    df["C_MID"] = 0.5 * (df["C_BID"] + df["C_ASK"])
    df["P_MID"] = 0.5 * (df["P_BID"] + df["P_ASK"])
    df["DIFF"] = (df["C_MID"] - df["P_MID"]).abs()
    T = float(df["DTE"].iloc[0]) / 365.25
    j = df["DIFF"].idxmin()
    Kstar = float(df.loc[j, "STRIKE"])
    parity = float(df.loc[j, "C_MID"] - df.loc[j, "P_MID"])
    F = Kstar + parity * np.exp(r * T)
    return F, Kstar, T


def build_smile(options_df, rates_df, trade_date, dte=None, expire_date=None,
                min_bid=0.05, otm_only=True):
    """Select one (trade_date, expiry) slice and assemble a clean smile.

    Exactly one of `dte` or `expire_date` should be specified.  When `dte`
    is given we pick the expiry whose DTE is closest to `dte`.

    Filter / construction steps
    ---------------------------
    1. select the slice,
    2. compute T in years, interpolate r from the yield curve,
    3. extract implied forward F via put-call parity,
    4. keep OTM side (puts for K<F, calls for K>F) if `otm_only`,
    5. drop rows with bid < `min_bid` (illiquid) or NaN IV,
    6. return a tidy DataFrame with columns
       [STRIKE, K_over_F, T, F, r, SIGMA_MKT, MID, BID, ASK, OPTION].
    """
    trade_date = pd.Timestamp(trade_date)
    df = options_df[options_df["QUOTE_DATE"] == trade_date]
    if df.empty:
        raise ValueError(f"no options rows on {trade_date.date()}")

    if expire_date is not None:
        expire_date = pd.Timestamp(expire_date)
        sl = df[df["EXPIRE_DATE"] == expire_date].copy()
        if sl.empty:
            raise ValueError(f"no expiry {expire_date.date()} on {trade_date.date()}")
    else:
        if dte is None:
            raise ValueError("specify either `dte` or `expire_date`")
        avail = df["DTE"].unique()
        best = avail[np.argmin(np.abs(avail - dte))]
        sl = df[df["DTE"] == best].copy()

    T = float(sl["DTE"].iloc[0]) / 365.25
    r = interpolate_yield(rates_df, trade_date, T)
    F, Kstar, _ = extract_forward(sl, r)

    # pick which side (OTM) to trust
    sl["C_MID"] = 0.5 * (sl["C_BID"] + sl["C_ASK"])
    sl["P_MID"] = 0.5 * (sl["P_BID"] + sl["P_ASK"])

    rows = []
    for _, row in sl.iterrows():
        K = float(row["STRIKE"])
        use_call = K >= F if otm_only else True
        if use_call:
            mid, bid, ask, iv = row["C_MID"], row["C_BID"], row["C_ASK"], row["C_IV"]
            typ = "C"
        else:
            mid, bid, ask, iv = row["P_MID"], row["P_BID"], row["P_ASK"], row["P_IV"]
            typ = "P"
        if not np.isfinite(iv) or iv <= 0:
            continue
        if not np.isfinite(bid) or bid < min_bid:
            continue
        if not np.isfinite(ask) or ask <= bid:
            continue
        rows.append({
            "STRIKE": K, "K_over_F": K / F, "T": T, "F": F, "r": r,
            "SIGMA_MKT": iv, "MID": mid, "BID": bid, "ASK": ask, "OPTION": typ,
        })
    out = pd.DataFrame(rows).sort_values("STRIKE").reset_index(drop=True)
    out.attrs.update({"trade_date": trade_date, "expire_dte": T * 365.25, "F": F, "r": r})
    return out
