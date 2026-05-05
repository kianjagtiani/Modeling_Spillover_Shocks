"""
Generate two presentation charts that fill gaps in btc_volatility_research.ipynb:

  1. Strategy equity curve + drawdown                    -> results/strategy_equity_drawdown.png
  2. Feature importance for HAR-LGBM Hybrid              -> results/feature_importance.png

Run:
    python add_charts_strategy_shap.py

Outputs PNGs at 150 DPI, dark theme matching the notebook palette.
After running, drop into the notebook between sections with:

    from IPython.display import Image
    Image('results/feature_importance.png')          # between section 3 and 4
    Image('results/strategy_equity_drawdown.png')    # inside section 8

Self-contained. Requires: pandas, numpy, matplotlib, pyarrow, yfinance.
SHAP and LightGBM are optional - the script degrades gracefully without them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
RESULTS  = ROOT / "results"
HYBRID   = RESULTS / "preds_har_lgbm_hybrid_h1.parquet"
GRANGER  = RESULTS / "lead_lag_granger.csv"

OUT_STRATEGY = RESULTS / "strategy_equity_drawdown.png"
OUT_SHAP     = RESULTS / "feature_importance.png"


# ── Theme (matches the notebook) ──────────────────────────────────────────────
ORANGE, BLUE, GREEN, RED, GRAY, LTGRAY = (
    "#f7931a", "#3498db", "#2ecc71", "#e74c3c", "#6c8ead", "#ccd6e0"
)
plt.rcParams.update({
    "figure.facecolor": "#0d1b2a", "axes.facecolor": "#0d1b2a",
    "axes.edgecolor":   "#6c8ead", "axes.labelcolor":  "#ccd6e0",
    "xtick.color":      "#ccd6e0", "ytick.color":      "#ccd6e0",
    "text.color":       "#ffffff", "grid.color":       "#1a2d42",
    "grid.linewidth":   0.6,       "font.size":        11,
    "axes.titlesize":   13,        "axes.titleweight": "bold",
    "legend.framealpha":0.3,       "legend.facecolor": "#0d1b2a",
})


# ──────────────────────────────────────────────────────────────────────────────
# CHART 1 - Strategy equity curve + drawdown
# ──────────────────────────────────────────────────────────────────────────────

def load_btc_returns(start: str, end: str) -> pd.Series:
    """Daily log returns for BTC-USD over the OOS window."""
    import yfinance as yf
    df = yf.download("BTC-USD", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("yfinance returned no BTC-USD data - check internet connection")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].astype(float)
    ret = np.log(close / close.shift(1)).dropna()
    ret.index = pd.to_datetime(ret.index, utc=True).normalize()
    ret.name = "btc_ret"
    return ret


def run_strategy(btc_ret: pd.Series, pred_log_rv: pd.Series,
                 target_vol: float = 0.02, max_pos: float = 1.5,
                 min_pos: float = 0.10, tx_cost: float = 0.001
                 ) -> dict[str, pd.Series]:
    """Reproduces the four strategies from src/backtest.py.
    Predicted *daily* vol = exp(pred_log_rv) / sqrt(365).
    Returns dict of cumulative-return series, one per strategy."""
    idx = btc_ret.index.intersection(pred_log_rv.index)
    r   = btc_ret.loc[idx]
    pv  = np.exp(pred_log_rv.loc[idx]) / np.sqrt(365)

    # Buy & Hold
    bh = r.copy()

    # Inverse-vol sizing (yesterday's prediction sizes today)
    pos_iv = (target_vol / pv).clip(min_pos, max_pos).shift(1).fillna(1.0)
    iv_ret = pos_iv * r - pos_iv.diff().abs().fillna(0) * tx_cost

    # Regime filter
    q33, q67 = pv.quantile(0.33), pv.quantile(0.67)
    pos_rf = pd.Series(1.0, index=idx)
    pos_rf[pv < q33] = 1.3
    pos_rf[pv > q67] = 0.5
    pos_rf = pos_rf.clip(min_pos, max_pos).shift(1).fillna(1.0)
    rf_ret = pos_rf * r - pos_rf.diff().abs().fillna(0) * tx_cost

    # Combined
    pos_co = ((target_vol / pv) * pos_rf).clip(min_pos, max_pos).shift(1).fillna(1.0)
    co_ret = pos_co * r - pos_co.diff().abs().fillna(0) * tx_cost

    return {
        "Buy & Hold":     (1 + bh).cumprod(),
        "Inverse-Vol":    (1 + iv_ret).cumprod(),
        "Regime Filter":  (1 + rf_ret).cumprod(),
        "Combined":       (1 + co_ret).cumprod(),
    }


def drawdown(equity: pd.Series) -> pd.Series:
    return (equity / equity.cummax() - 1) * 100


def chart_strategy(equity: dict[str, pd.Series]) -> None:
    """Stacked equity-curve (top) + drawdown (bottom)."""
    colors = {
        "Buy & Hold":    GRAY,
        "Inverse-Vol":   BLUE,
        "Regime Filter": GREEN,
        "Combined":      ORANGE,
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("Strategy Performance: Equity Curve & Drawdown   "
                 "(2022-2026, 0.1% tx cost, OOS)", fontsize=13)

    # ── Top: equity curves ───────────────────────────────────────────────────
    ax = axes[0]
    for name, eq in equity.items():
        c = colors[name]
        lw = 2.2 if name == "Combined" else 1.4
        alpha = 1.0 if name == "Combined" else 0.85
        ax.plot(eq.index, eq.values, color=c, linewidth=lw, alpha=alpha, label=name)
        # Endpoint label
        end_ret = (eq.iloc[-1] - 1) * 100
        ax.annotate(f"{end_ret:+.1f}%",
                    xy=(eq.index[-1], eq.iloc[-1]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=11, color=c, va="center", weight="bold")

    ax.axhline(1.0, color=LTGRAY, linewidth=0.6, linestyle=":")
    ax.set_ylabel("Cumulative Return  (1.0 = starting capital)")
    ax.legend(loc="lower center", fontsize=10, ncol=4,
              bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax.grid(alpha=0.25)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{v:.1f}x" if v >= 1 else f"{v:.2f}x"))

    # ── Bottom: drawdown ─────────────────────────────────────────────────────
    ax2 = axes[1]
    for name, eq in equity.items():
        dd = drawdown(eq)
        c = colors[name]
        lw = 1.8 if name == "Combined" else 1.0
        alpha = 0.95 if name == "Combined" else 0.7
        if name == "Combined":
            ax2.fill_between(dd.index, dd.values, 0, color=c, alpha=0.18)
        ax2.plot(dd.index, dd.values, color=c, linewidth=lw, alpha=alpha, label=name)

    ax2.axhline(0, color=LTGRAY, linewidth=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("")
    ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    # Annotate worst drawdown for B&H and Combined
    for name in ("Buy & Hold", "Combined"):
        dd = drawdown(equity[name])
        worst_t = dd.idxmin()
        ax2.annotate(f"{name}: {dd.min():.1f}%",
                     xy=(worst_t, dd.min()),
                     xytext=(0, -14), textcoords="offset points",
                     fontsize=8, color=colors[name], ha="center")

    plt.tight_layout()
    plt.savefig(OUT_STRATEGY, dpi=150, bbox_inches="tight",
                facecolor="#0d1b2a", edgecolor="none")
    plt.close(fig)
    print(f"  saved {OUT_STRATEGY}")


# ──────────────────────────────────────────────────────────────────────────────
# CHART 2 - Feature importance (3-tier fallback)
# ──────────────────────────────────────────────────────────────────────────────

def feature_importance_via_lightgbm(actual: pd.Series) -> pd.DataFrame | None:
    """Train a single LightGBM on classic HAR features + lagged actual log-RV.
    This is a *surrogate* for the walk-forward Hybrid model - the goal is to
    show which feature *families* dominate, not to reproduce the full model.
    Returns DataFrame with columns [feature, importance, family] or None."""
    try:
        import lightgbm as lgb
    except Exception as e:
        print(f"  lightgbm import failed ({type(e).__name__}: {e})")
        return None

    y = actual.copy()
    y.index = pd.to_datetime(y.index, utc=True).normalize()

    # HAR-style own-history features (these are what the hybrid uses)
    df = pd.DataFrame({"y": y})
    df["btc_rv_lag1"]   = y.shift(1)
    df["btc_rv_lag5"]   = y.shift(1).rolling(5).mean()
    df["btc_rv_lag22"]  = y.shift(1).rolling(22).mean()
    df["btc_rv_chg1d"]  = y.shift(1) - y.shift(2)
    df["btc_rv_chg5d"]  = y.shift(1) - y.shift(6)
    df["btc_rv_vol22"]  = y.shift(1).rolling(22).std()

    # Cross-asset proxies via yfinance (same families as the notebook)
    try:
        import yfinance as yf
        start = (df.index[0] - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
        end   = (df.index[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        cross = yf.download(
            ["ETH-USD", "SOL-USD", "BNB-USD", "^VIX", "^GSPC", "GC=F", "DX-Y.NYB"],
            start=start, end=end, progress=False, auto_adjust=False,
        )["Close"]
        cross.index = pd.to_datetime(cross.index, utc=True).normalize()
        cross = cross.reindex(df.index).ffill()

        for col, label in [
            ("ETH-USD",   "eth_vol_5d"),
            ("SOL-USD",   "sol_vol_5d"),
            ("BNB-USD",   "bnb_vol_5d"),
        ]:
            r = np.log(cross[col] / cross[col].shift(1))
            df[label] = r.rolling(5).std() * np.sqrt(365)
        df["vix_level"]   = cross["^VIX"]
        df["sp500_ret"]   = np.log(cross["^GSPC"] / cross["^GSPC"].shift(1))
        df["gold_ret"]    = np.log(cross["GC=F"] / cross["GC=F"].shift(1))
        df["dxy_ret"]     = np.log(cross["DX-Y.NYB"] / cross["DX-Y.NYB"].shift(1))
    except Exception as e:
        print(f"  cross-asset fetch failed: {e}  -- using own-history features only")

    df = df.dropna()
    if len(df) < 200:
        return None

    X = df.drop(columns=["y"])
    y_train = df["y"]

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        num_leaves=31, min_child_samples=20, random_state=0, verbosity=-1,
    )
    model.fit(X, y_train)

    imp = model.booster_.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": X.columns, "importance": imp})
    imp_df["importance"] = imp_df["importance"] / imp_df["importance"].sum() * 100

    # Try SHAP on top of LightGBM for a richer chart
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X.tail(min(2000, len(X))))
        mean_abs = np.abs(sv).mean(axis=0)
        imp_df["shap"] = mean_abs / mean_abs.sum() * 100
    except Exception:
        imp_df["shap"] = np.nan

    imp_df["family"] = imp_df["feature"].map(_family_of)
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def feature_importance_via_granger() -> pd.DataFrame:
    """Last-resort fallback: rank EXTERNAL features by Granger F-statistic.
    Note: BTC own-history features are the model's baseline (untested in
    Granger), so they don't appear here. The chart should be labelled
    "external-feature importance" to avoid misleading the audience."""
    g = pd.read_csv(GRANGER)
    df = pd.DataFrame({
        "feature":    g["feature"],
        "importance": g["f_stat"] / g["f_stat"].sum() * 100,
        "shap":       np.nan,
        "family":     g["feature"].map(_family_of),
    })
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _family_of(feat: str) -> str:
    f = feat.lower()
    # Cross-asset takes precedence: a feature like "btc_eth_corr_10d" is a
    # cross-asset signal, not BTC own history, even though it contains "btc".
    if any(k in f for k in ("eth", "sol", "bnb", "ada",
                            "dot", "avax", "uni", "aave", "mkr")):
        return "Crypto cross-asset"
    if any(k in f for k in ("vix", "sp500", "ndx", "dji", "xlk", "xlf", "soxx")):
        return "Equities / VIX"
    if any(k in f for k in ("gold", "wti", "brent", "copper", "dxy", "tnx")):
        return "Macro / Commodities"
    if "fg" in f:                                            return "Sentiment"
    if "btc" in f or f.startswith("rv_") or "_rv_" in f:    return "BTC own history"
    return "Other"


FAMILY_COLORS = {
    "BTC own history":     ORANGE,
    "Crypto cross-asset":  GREEN,
    "Equities / VIX":      BLUE,
    "Macro / Commodities": "#9b59b6",
    "Sentiment":           "#f1c40f",
    "Other":               GRAY,
}


def chart_feature_importance(imp_df: pd.DataFrame, source: str,
                              external_only: bool = False) -> None:
    """Two panels: top-N bar chart, family-aggregated donut.
    `external_only`=True adds a caveat: BTC own-history features are not in this view."""
    top = imp_df.head(15).iloc[::-1]   # reverse for horizontal bar (largest on top)
    family_totals = (
        imp_df.groupby("family")["importance"].sum()
              .sort_values(ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    title_main = ("Which External Signals Predict BTC Volatility?"
                  if external_only
                  else "What Drives the HAR-LGBM Hybrid Forecast?")
    subtitle = (f"{source}   |   BTC own-history is the baseline (not shown)"
                if external_only else source)
    fig.suptitle(f"{title_main}\n{subtitle}", fontsize=12)

    # ── Left: top-15 features ────────────────────────────────────────────────
    ax = axes[0]
    bar_colors = [FAMILY_COLORS[f] for f in top["family"]]
    bars = ax.barh(top["feature"], top["importance"],
                   color=bar_colors, alpha=0.88, edgecolor="none")
    for bar, v in zip(bars, top["importance"]):
        ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9, color=LTGRAY)
    ax.set_xlabel("Relative importance (%)")
    ax.set_title("Top 15 Features", fontsize=11)
    ax.grid(axis="x", alpha=0.3)

    # ── Right: family donut ──────────────────────────────────────────────────
    ax2 = axes[1]
    fam_colors = [FAMILY_COLORS[f] for f in family_totals.index]
    wedges, _ = ax2.pie(
        family_totals.values, colors=fam_colors,
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="#0d1b2a", linewidth=2),
    )
    ax2.set_title("Importance by Feature Family", fontsize=11)

    # Center label: when we have BTC own-history, highlight its share.
    # In external-only mode, highlight the dominant cross-asset family instead.
    if "BTC own history" in family_totals.index:
        center_pct = family_totals["BTC own history"]
        center_lbl = "BTC own\nhistory"
    else:
        top_fam = family_totals.index[0]
        center_pct = family_totals.iloc[0]
        center_lbl = top_fam.replace(" ", "\n")
    ax2.text(0, 0.05, f"{center_pct:.0f}%",
             ha="center", va="center", fontsize=22, color=ORANGE, weight="bold")
    ax2.text(0, -0.18, center_lbl,
             ha="center", va="center", fontsize=9, color=LTGRAY)

    legend_handles = [
        mpatches.Patch(color=FAMILY_COLORS[f], label=f"{f}  ({family_totals[f]:.1f}%)")
        for f in family_totals.index
    ]
    ax2.legend(handles=legend_handles, loc="center left",
               bbox_to_anchor=(1.05, 0.5), fontsize=9, frameon=False)

    plt.tight_layout()
    plt.savefig(OUT_SHAP, dpi=150, bbox_inches="tight",
                facecolor="#0d1b2a", edgecolor="none")
    plt.close(fig)
    print(f"  saved {OUT_SHAP}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not HYBRID.exists():
        sys.exit(f"missing predictions file: {HYBRID}")

    try:
        preds = pd.read_parquet(HYBRID, engine="pyarrow")
    except Exception:
        preds = pd.read_parquet(HYBRID)
    preds.index = pd.to_datetime(preds.index, utc=True).normalize()
    preds = preds.iloc[:-1]   # drop the unrealized-future last row, matches backtest.py

    # ── Strategy chart ───────────────────────────────────────────────────────
    print("Building strategy equity / drawdown chart...")
    start = (preds.index[0] - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end   = (preds.index[-1] + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    btc_ret = load_btc_returns(start, end)
    equity  = run_strategy(btc_ret, preds["predicted"])
    chart_strategy(equity)

    # Print summary table for the teammate
    print(f"\n  {'Strategy':<16} {'Total':>9} {'CAGR':>8} {'MaxDD':>8} {'Sharpe':>8}")
    print(f"  {'-'*52}")
    for name, eq in equity.items():
        ret = eq.pct_change().dropna()
        n = len(ret)
        total = (eq.iloc[-1] - 1) * 100
        cagr  = ((eq.iloc[-1]) ** (365 / n) - 1) * 100
        dd    = drawdown(eq).min()
        sh    = (ret.mean() / ret.std() * np.sqrt(365)) if ret.std() > 0 else 0
        print(f"  {name:<16} {total:>8.1f}% {cagr:>7.1f}% {dd:>7.1f}% {sh:>8.2f}")

    # ── Feature importance chart (with 3-tier fallback) ──────────────────────
    print("\nBuilding feature-importance chart...")
    imp = feature_importance_via_lightgbm(preds["actual"])
    if imp is not None:
        source = ("SHAP values (LightGBM surrogate)" if imp["shap"].notna().any()
                  else "LightGBM gain importance (surrogate model)")
        if imp["shap"].notna().any():
            imp = imp.copy()
            imp["importance"] = imp["shap"]
            imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        print("  lightgbm unavailable -- falling back to Granger F-statistic ranking")
        imp = feature_importance_via_granger()
        source = "Granger F-statistic (proxy)"

    print(f"\n  Top 5 features ({source}):")
    for _, r in imp.head(5).iterrows():
        print(f"    {r['feature']:<22}  {r['importance']:>5.1f}%   [{r['family']}]")

    external_only = "Granger" in source
    chart_feature_importance(imp, source, external_only=external_only)

    print("\nDone. To embed in the notebook:")
    print("    from IPython.display import Image")
    print(f"    Image('{OUT_SHAP.relative_to(ROOT)}')           # between sec 3 and 4")
    print(f"    Image('{OUT_STRATEGY.relative_to(ROOT)}')   # inside sec 8")


if __name__ == "__main__":
    main()
