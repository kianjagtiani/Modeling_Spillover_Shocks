"""
Lead-lag and cross-asset spillover analysis for BTC volatility.

Research question: which assets predict BTC realized volatility,
at what lag, and in which market regimes?

Functions:
  granger_causality_table  — AIC-selected Granger causality for each feature
  cross_correlation_profile — CCF at lags -15..+15 with significance
  rolling_lead_lag          — 90-day rolling optimal lag detection
  conditional_granger_by_regime — regime-split Granger tests
  spillover_table           — VAR-based FEVD (Diebold-Yilmaz style)
  run_lead_lag_analysis     — orchestrates all analyses, saves reports
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR   = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Asset groups used in analysis
TRADITIONAL_COLS = [
    "vix_level", "log_vix", "vix_5d_chg",
    "sp500_ret_1d", "sp500_vol_5d", "sp500_vol_22d",
    "gold_ret_1d", "gold_vol_5d",
    "dxy_ret_1d", "dxy_vol_5d",
    "tnx_level", "tnx_5d_chg",
]
CRYPTO_COLS = [
    "eth_ret_1d", "eth_vol_5d",
    "sol_ret_1d", "sol_vol_5d",
    "bnb_ret_1d", "bnb_vol_5d",
    "btc_eth_corr_10d",
]
SENTIMENT_COLS = ["fg_lag1", "fg_7d_chg", "fg_extreme_fear", "fg_extreme_greed"]
HAR_COLS = ["rv_1d", "rv_5d", "rv_22d"]

# VAR key variables for spillover analysis
VAR_VARS = ["log_rv", "log_vix", "sp500_vol_5d", "eth_vol_5d", "sol_vol_5d",
            "dxy_ret_1d", "tnx_5d_chg", "gold_ret_1d"]


# ── Stationarity helper ───────────────────────────────────────────────────────

def _make_stationary(series: pd.Series, name: str = "") -> pd.Series:
    """First-difference if ADF rejects stationarity."""
    result = adfuller(series.dropna(), autolag="AIC")
    if result[1] > 0.05:
        return series.diff().dropna()
    return series


# ── 1. Granger Causality Table ────────────────────────────────────────────────

def granger_causality_table(
    features_df: pd.DataFrame,
    target_col: str = "log_rv",
    max_lag: int = 10,
    candidate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Test each feature for Granger causality with respect to target_col.
    Reports the lag with the highest F-statistic (most evidence of predictability).

    Returns DataFrame sorted by p-value (ascending = most significant first).
    """
    if candidate_cols is None:
        candidate_cols = TRADITIONAL_COLS + CRYPTO_COLS + SENTIMENT_COLS
    candidate_cols = [c for c in candidate_cols if c in features_df.columns and c != target_col]

    y = features_df[target_col].dropna()
    records = []

    for col in candidate_cols:
        pair = features_df[[target_col, col]].dropna()
        if len(pair) < max_lag * 5 + 10:
            continue

        # Make both series stationary before testing
        y_stat = _make_stationary(pair[target_col], target_col)
        x_stat = _make_stationary(pair[col], col)
        data   = pd.concat([y_stat, x_stat], axis=1).dropna()

        if len(data) < max_lag * 5:
            continue

        try:
            gc_tests = grangercausalitytests(data.values, maxlag=max_lag, verbose=False)
        except Exception:
            continue

        # Find lag with highest F-stat (strongest evidence)
        best_lag, best_f, best_p = 1, -np.inf, 1.0
        for lag, res_list in gc_tests.items():
            f_stat = res_list[0]["ssr_ftest"][0]
            p_val  = res_list[0]["ssr_ftest"][1]
            if f_stat > best_f:
                best_f, best_p, best_lag = f_stat, p_val, lag

        # Also report F-stat at lag 1 (most directly usable)
        f_lag1 = gc_tests[1][0]["ssr_ftest"][0]
        p_lag1 = gc_tests[1][0]["ssr_ftest"][1]

        records.append({
            "feature":         col,
            "optimal_lag":     best_lag,
            "f_stat":          round(best_f, 3),
            "p_value":         round(best_p, 4),
            "significant_5pct": best_p < 0.05,
            "f_stat_lag1":     round(f_lag1, 3),
            "p_value_lag1":    round(p_lag1, 4),
            "n_obs":           len(data),
        })

    result = pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)
    return result


# ── 2. Cross-Correlation Profile ──────────────────────────────────────────────

def cross_correlation_profile(
    series_a: pd.Series,
    series_b: pd.Series,
    max_lag: int = 15,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """
    Compute CCF at lags -max_lag..+max_lag.
    Positive lag k: series_a at t-k correlates with series_b at t (a leads b).
    Returns dict with the peak |correlation| lag and its significance.
    """
    a = series_a.dropna()
    b = series_b.dropna()
    common = a.index.intersection(b.index)
    a, b   = a.loc[common], b.loc[common]

    n    = len(a)
    lags = list(range(-max_lag, max_lag + 1))
    corrs, pvals = [], []

    for lag in lags:
        if lag == 0:
            shifted_a = a
            shifted_b = b
        elif lag > 0:
            # a leads b: a at t-lag with b at t
            shifted_a = a.iloc[:-lag]
            shifted_b = b.iloc[lag:]
        else:
            # b leads a: b at t-|lag| with a at t
            shifted_a = a.iloc[-lag:]
            shifted_b = b.iloc[:lag]

        if len(shifted_a) < 20:
            corrs.append(np.nan)
            pvals.append(np.nan)
            continue

        r, p = stats.pearsonr(shifted_a.values, shifted_b.values)
        corrs.append(r)
        pvals.append(p)

    # Find peak |correlation|
    abs_corrs    = [abs(c) if not np.isnan(c) else 0 for c in corrs]
    peak_idx     = int(np.argmax(abs_corrs))
    peak_lag     = lags[peak_idx]
    peak_corr    = corrs[peak_idx]
    peak_pval    = pvals[peak_idx]

    if peak_lag > 0:
        direction = f"{label_a} leads {label_b} by {peak_lag} day(s)"
    elif peak_lag < 0:
        direction = f"{label_b} leads {label_a} by {abs(peak_lag)} day(s)"
    else:
        direction = "contemporaneous (lag 0)"

    return {
        "lags":       lags,
        "correlations": corrs,
        "p_values":   pvals,
        "peak_lag":   peak_lag,
        "peak_corr":  round(peak_corr, 4) if peak_corr is not None else None,
        "peak_pval":  round(peak_pval, 4) if peak_pval is not None else None,
        "direction":  direction,
        "significant_5pct": (peak_pval is not None and peak_pval < 0.05),
    }


# ── 3. Rolling Lead-Lag Detection ─────────────────────────────────────────────

def rolling_lead_lag(
    features_df: pd.DataFrame,
    target_col: str,
    asset_col: str,
    window: int = 90,
    max_lag: int = 10,
) -> pd.DataFrame:
    """
    Rolling window cross-correlation to detect time-varying lead-lag structure.
    Returns per-day: optimal lag, correlation, vol regime label.
    """
    pair = features_df[[target_col, asset_col, "rv_pct_60d"]].dropna()
    dates, opt_lags, opt_corrs, regimes = [], [], [], []

    for i in range(window, len(pair)):
        window_df = pair.iloc[i - window:i]
        y_w = window_df[target_col]
        x_w = window_df[asset_col]

        best_lag, best_corr = 0, 0.0
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                r, _ = stats.pearsonr(y_w, x_w)
            elif lag > 0:
                r, _ = stats.pearsonr(y_w.iloc[lag:], x_w.iloc[:-lag])
            else:
                r, _ = stats.pearsonr(y_w.iloc[:lag], x_w.iloc[-lag:])
            if abs(r) > abs(best_corr):
                best_corr, best_lag = r, lag

        pct = window_df["rv_pct_60d"].iloc[-1]
        regime = "Low" if pct < 0.33 else ("High" if pct > 0.67 else "Normal")

        dates.append(pair.index[i])
        opt_lags.append(best_lag)
        opt_corrs.append(round(best_corr, 4))
        regimes.append(regime)

    return pd.DataFrame({
        "date":         dates,
        "optimal_lag":  opt_lags,
        "correlation":  opt_corrs,
        "regime":       regimes,
    }).set_index("date")


# ── 4. Regime-Conditional Granger Causality ───────────────────────────────────

def conditional_granger_by_regime(
    features_df: pd.DataFrame,
    target_col: str = "log_rv",
    candidate_cols: list[str] | None = None,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Run Granger causality tests within each vol regime separately.
    Identifies whether VIX/equity signals are regime-specific.
    Returns DataFrame: (feature, regime) → F-stat, p-value.
    """
    if candidate_cols is None:
        candidate_cols = TRADITIONAL_COLS + CRYPTO_COLS
    candidate_cols = [c for c in candidate_cols if c in features_df.columns]

    if "rv_pct_60d" not in features_df.columns:
        pct = features_df[target_col].rank(pct=True)
    else:
        pct = features_df["rv_pct_60d"]

    masks = {
        "Low":    pct < 0.33,
        "Normal": (pct >= 0.33) & (pct <= 0.67),
        "High":   pct > 0.67,
    }

    records = []
    for col in candidate_cols:
        for regime, mask in masks.items():
            subset = features_df.loc[mask, [target_col, col]].dropna()
            if len(subset) < max_lag * 4 + 5:
                continue
            try:
                gc = grangercausalitytests(subset.values, maxlag=max_lag, verbose=False)
                f_stat = gc[1][0]["ssr_ftest"][0]
                p_val  = gc[1][0]["ssr_ftest"][1]
                records.append({
                    "feature": col,
                    "regime":  regime,
                    "f_stat":  round(f_stat, 3),
                    "p_value": round(p_val, 4),
                    "significant_5pct": p_val < 0.05,
                    "n_obs":   len(subset),
                })
            except Exception:
                pass

    return pd.DataFrame(records).sort_values(["feature", "regime"]).reset_index(drop=True)


# ── 5. Spillover Table (Diebold-Yilmaz style) ─────────────────────────────────

def spillover_table(
    features_df: pd.DataFrame,
    variables: list[str] | None = None,
    lags: int = 5,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    VAR-based forecast error variance decomposition (Diebold & Yilmaz 2012).
    Returns a (n_vars × n_vars) matrix where entry [i, j] is the % of
    variable i's forecast variance attributable to shocks in variable j.

    The row 'log_rv' shows which assets explain BTC realized vol forecasts.
    """
    if variables is None:
        variables = [v for v in VAR_VARS if v in features_df.columns]

    data = features_df[variables].dropna()

    # Make all series stationary
    data_stat = data.copy()
    for col in variables:
        adf_p = adfuller(data[col], autolag="AIC")[1]
        if adf_p > 0.05:
            data_stat[col] = data[col].diff()
    data_stat = data_stat.dropna()

    try:
        model  = VAR(data_stat)
        fitted = model.fit(maxlags=lags, ic="aic", verbose=False)
        fevd   = fitted.fevd(periods=horizon)
        # fevd.decomp: shape (horizon, n_vars, n_vars) — [h, i, j] = contribution of j to i at h
        avg_fevd = fevd.decomp[-1]  # Use final horizon
        result   = pd.DataFrame(
            avg_fevd * 100,
            index=variables,
            columns=variables,
        ).round(2)
        return result
    except Exception as e:
        print(f"  [spillover] VAR failed: {e}")
        return pd.DataFrame()


# ── 6. Orchestrator ───────────────────────────────────────────────────────────

def run_lead_lag_analysis(data_path: Path | None = None) -> dict:
    """
    Load features.parquet, run all lead-lag analyses, print + save results.
    Returns dict of results DataFrames.
    """
    path = data_path or (PROCESSED_DIR / "features.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {path}")

    print(f"Loading feature matrix from {path.name}...")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"  {len(df):,} rows × {len(df.columns)} cols, "
          f"{df.index.min().date()} → {df.index.max().date()}\n")

    results = {}

    # ── Granger causality ──────────────────────────────────────────────────────
    print("=" * 68)
    print("GRANGER CAUSALITY: WHICH ASSETS PREDICT BTC VOLATILITY?")
    print("=" * 68)
    gc_table = granger_causality_table(df, max_lag=10)
    results["granger"] = gc_table

    sig = gc_table[gc_table["significant_5pct"]]
    print(f"\n{'Feature':<30} {'Opt Lag':>8} {'F-stat':>8} {'p-value':>9} {'Sig?':>5}")
    print("-" * 65)
    for _, row in gc_table.head(20).iterrows():
        sig_flag = "  ***" if row["p_value"] < 0.001 else (" **" if row["p_value"] < 0.01
                   else (" *" if row["significant_5pct"] else ""))
        print(f"{row['feature']:<30} {row['optimal_lag']:>8} "
              f"{row['f_stat']:>8.3f} {row['p_value']:>9.4f}{sig_flag}")

    print(f"\n  → {len(sig)}/{len(gc_table)} features Granger-cause BTC vol (p < 0.05)")

    # ── Cross-correlation profiles ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("CROSS-CORRELATION PROFILES (peak lag and direction)")
    print("=" * 68)
    ccf_records = []
    key_assets = [c for c in TRADITIONAL_COLS + CRYPTO_COLS if c in df.columns]
    for col in key_assets[:15]:
        prof = cross_correlation_profile(
            df[col], df["log_rv"],
            label_a=col, label_b="log_rv"
        )
        ccf_records.append({
            "feature":    col,
            "peak_lag":   prof["peak_lag"],
            "peak_corr":  prof["peak_corr"],
            "p_value":    prof["peak_pval"],
            "direction":  prof["direction"],
            "significant": prof["significant_5pct"],
        })
        if prof["significant_5pct"]:
            print(f"  {col:<30}  lag={prof['peak_lag']:>3}  r={prof['peak_corr']:>6.3f}  "
                  f"p={prof['peak_pval']:.4f}  {prof['direction']}")

    results["ccf"] = pd.DataFrame(ccf_records)

    # ── Regime-conditional Granger ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("REGIME-CONDITIONAL GRANGER CAUSALITY")
    print("(Does VIX lead BTC more in high-vol regimes?)")
    print("=" * 68)
    regime_gc = conditional_granger_by_regime(
        df,
        candidate_cols=["log_vix", "vix_5d_chg", "sp500_vol_5d", "eth_vol_5d",
                        "sol_vol_5d", "dxy_ret_1d", "tnx_5d_chg"]
    )
    results["regime_granger"] = regime_gc

    key_features = ["log_vix", "sp500_vol_5d", "eth_vol_5d"]
    for feat in key_features:
        sub = regime_gc[regime_gc["feature"] == feat]
        if sub.empty:
            continue
        print(f"\n  {feat}:")
        for _, row in sub.iterrows():
            sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01
                  else ("*" if row["significant_5pct"] else "n.s."))
            print(f"    {row['regime']:>7} regime: F={row['f_stat']:.3f}  "
                  f"p={row['p_value']:.4f} [{sig}]  n={row['n_obs']}")

    # ── Spillover table ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("DIEBOLD-YILMAZ SPILLOVER TABLE")
    print("(% of BTC vol forecast variance explained by each asset)")
    print("=" * 68)
    spill = spillover_table(df)
    results["spillover"] = spill

    if "log_rv" in spill.index:
        btc_row = spill.loc["log_rv"].sort_values(ascending=False)
        print(f"\n  BTC vol variance decomposition (h=10 step):")
        for var, pct in btc_row.items():
            if pct > 1:
                print(f"    {var:<30}  {pct:>6.1f}%")

    # ── Save outputs ──────────────────────────────────────────────────────────
    gc_path    = RESULTS_DIR / "lead_lag_granger.csv"
    ccf_path   = RESULTS_DIR / "lead_lag_ccf.csv"
    report_path = RESULTS_DIR / "lead_lag_report.txt"

    gc_table.to_csv(gc_path, index=False)
    results["ccf"].to_csv(ccf_path, index=False)

    report_lines = [
        "BTC Volatility Lead-Lag Analysis",
        f"Data: {df.index.min().date()} → {df.index.max().date()}  ({len(df):,} days)",
        "=" * 68,
        f"\nGranger causality: {len(sig)} significant predictors (p<0.05)",
        "\nTop 10 Granger-causal features:",
    ] + [f"  {row['feature']:<28} lag={row['optimal_lag']:>2}  p={row['p_value']:.4f}"
         for _, row in gc_table.head(10).iterrows()]

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nSaved:")
    print(f"  {gc_path.name}")
    print(f"  {ccf_path.name}")
    print(f"  {report_path.name}")

    return results


if __name__ == "__main__":
    run_lead_lag_analysis()
