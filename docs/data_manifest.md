# Data Manifest

This document specifies the raw input data required to reproduce the
empirical results in this repository end-to-end. **None of the raw data
is tracked in git** (the `Data/` and `cache/` directories are
gitignored) because:

- the option chains are licensed by optionsdx.com and cannot be
  redistributed under their terms of use;
- the unfiltered raw text totals roughly 4.4 GB, well beyond a
  comfortable git repo size;
- the FRED yield series are trivial to re-download and the cached
  parquet artifacts are deterministic, so checking them in would mostly
  duplicate easily-regenerable bytes.

The layout below is the on-disk structure the code expects after you
download the data yourself.

```
Data/
├── DGS1MO.csv                     FRED 1-month CMT yield
├── DGS3MO.csv                     FRED 3-month CMT yield
├── DGS6MO.csv                     FRED 6-month CMT yield
├── DGS1.csv                       FRED 1-year CMT yield
├── DGS2.csv                       FRED 2-year CMT yield
├── spy/
│   ├── spy_eod_201401.txt         optionsdx EOD chain, Jan 2014
│   ├── spy_eod_201402.txt
│   └── ... (120 monthly files, 2014-01 through 2023-12)
└── qqq/
    ├── qqq_eod_201401.txt         optionsdx EOD chain, Jan 2014
    ├── qqq_eod_201402.txt
    └── ... (120 monthly files, 2014-01 through 2023-12)
```

## 1. FRED Treasury constant-maturity yields

| Series  | Tenor    | Source URL                                                       |
|---------|----------|------------------------------------------------------------------|
| DGS1MO  | 1 month  | <https://fred.stlouisfed.org/series/DGS1MO>                      |
| DGS3MO  | 3 months | <https://fred.stlouisfed.org/series/DGS3MO>                      |
| DGS6MO  | 6 months | <https://fred.stlouisfed.org/series/DGS6MO>                      |
| DGS1    | 1 year   | <https://fred.stlouisfed.org/series/DGS1>                        |
| DGS2    | 2 years  | <https://fred.stlouisfed.org/series/DGS2>                        |

For each series:

1. Open the FRED page above.
2. Click **Download → CSV**.
3. Set date range **2014-01-02 → 2023-12-29** (or download the full
   history; the loader drops out-of-range rows).
4. Save the file under `Data/` with the name shown in the table.

**Expected per file:** ~2,607 daily observations spanning 2014-01-02 to
2023-12-29 (after forward-filling US market holidays, which the code
handles automatically). Two columns: `observation_date` and the series
code (e.g. `DGS1MO`), with yields in percentage units (the code
decimalizes them on load).

## 2. optionsdx end-of-day option chains

| Ticker | Source                                       | Files                                          |
|--------|----------------------------------------------|------------------------------------------------|
| SPY    | <https://www.optionsdx.com/product/spy/>     | 120 monthly `*.txt` files, Jan 2014 - Dec 2023 |
| QQQ    | <https://www.optionsdx.com/product/qqq/>     | 120 monthly `*.txt` files, Jan 2014 - Dec 2023 |

Procedure:

1. Purchase or download the SPY and QQQ end-of-day option chain
   archives from optionsdx.com.
2. Unzip into `Data/spy/` and `Data/qqq/` respectively, so each file is
   named `spy_eod_YYYYMM.txt` / `qqq_eod_YYYYMM.txt`.

**File format.** Comma-separated with a leading bracketed header row,
roughly 70,000–100,000 quotes per file. The 33 raw columns include
`[QUOTE_DATE]`, `[EXPIRE_DATE]`, `[DTE]`, `[UNDERLYING_LAST]`,
`[STRIKE]`, `[C_BID]`, `[C_ASK]`, `[C_IV]`, `[P_BID]`, `[P_ASK]`,
`[P_IV]`, `[STRIKE_DISTANCE_PCT]`, plus all four Greeks per side.
`src/data_loader.py` strips the brackets, retains 14 of these columns,
and applies the liquidity filters described below.

**Expected post-filter row counts** (after `|K/F-1| ≤ 0.30`,
`DTE ∈ [5, 365]`, bid ≥ 5¢):

| Ticker | Filtered rows |
|--------|---------------|
| SPY    | 5,281,485     |
| QQQ    | 3,547,089     |
| Total  | 8,828,574     |

If your row counts differ by more than ~1% from the above, double-check
the filter parameters in `notebooks/05_data_loading.ipynb` and the
DTE/moneyness thresholds in `src/data_loader.py`.

## 3. Cached intermediate artifacts

After running `notebooks/05_data_loading.ipynb` against the raw data
above, three parquet files appear under `cache/`:

| File                                  | Size  | Content                                                |
|---------------------------------------|-------|--------------------------------------------------------|
| `cache/spy_options_filtered.parquet`  | 74 MB | All 5,281,485 filtered SPY rows                        |
| `cache/qqq_options_filtered.parquet`  | 49 MB | All 3,547,089 filtered QQQ rows                        |
| `cache/calibration_grid.parquet`      | 4 KB  | 210 (ticker, trade_date, dte) calibration triples      |

After running `notebooks/06_calibration.ipynb` a fourth file appears:

| File                                  | Size  | Content                                                          |
|---------------------------------------|-------|------------------------------------------------------------------|
| `cache/calibration_results.parquet`   | 44 KB | 630 fitted SABR parameter rows: 210 smiles × 3 β ∈ {0, 0.5, 1}   |

## 4. End-to-end reproducibility checklist

```text
[ ]  Download 5 FRED CSVs into Data/
[ ]  Download SPY+QQQ EOD chains from optionsdx into Data/spy/ and Data/qqq/
[ ]  pip install -r requirements.txt
[ ]  Execute notebooks/05_data_loading.ipynb              (~2 min)
[ ]  Execute notebooks/06_calibration.ipynb               (~6 sec)
[ ]  Execute notebooks/07_model_comparison.ipynb          (~5 sec)
[ ]  python3 notebooks/export_figs_round2.py              regenerates fig11–17
[ ]  python3 notebooks/export_figs_round3.py              regenerates fig18–20
```

Total wall-clock time on a single laptop CPU thread: roughly **two
minutes** from raw data to all 20 figures, dominated by parsing the 240
optionsdx text files.

## 5. Sanity check after reproduction

After step 5 above (calibration), the printed summary in the notebook
should report:

```
Solver success rate: 100.0%
Median RMSE (bps of vol): 35.0
Median fn evals        : 9
Max RMSE seen (bps)    : 187.7
```

If your numbers differ materially, check:

1. The FRED yields are decimalized on load (e.g. 0.0525, not 5.25).
2. The filtered row counts match Section 2 above.
3. The calibration grid has 210 rows (35 quarterly trade dates × 2
   tickers × 3 DTEs).
