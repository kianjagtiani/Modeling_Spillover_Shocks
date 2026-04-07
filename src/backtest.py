"""
Trading strategy backtest using volatility forecasts.

Strategies tested:
  1. Buy & Hold BTC                    — benchmark
  2. Inverse-Vol Sizing (Risk Parity)  — size position = target_vol / predicted_vol
  3. Vol Regime Filter                 — scale exposure by vol regime (low/mid/high)
  4. Combined                          — inverse-vol sizing + regime filter

All strategies are long-only BTC, no leverage beyond 1.5x cap.
Transaction costs: 0.1% per trade (realistic for Binance spot).
No lookahead: predictions for day t use only data through day t-1.

Metrics reported:
  - Total return, annualized return
  - Sharpe ratio (annualized, risk-free = 0)
  - Sortino ratio
  - Max drawdown
  - Calmar ratio (ann. return / max drawdown)
  - Win rate (% of days with positive return)
  - VaR 5% and CVaR 5%
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    r = returns.dropna()
    cum   = (1 + r).cumprod()
    total = cum.iloc[-1] - 1
    n_days = len(r)
    ann_ret = (1 + total) ** (365 / n_days) - 1
    ann_vol = r.std() * np.sqrt(365)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = r[r < 0].std() * np.sqrt(365)
    sortino  = ann_ret / downside if downside > 0 else 0

    roll_max  = cum.cummax()
    drawdown  = (cum - roll_max) / roll_max
    max_dd    = drawdown.min()
    calmar    = ann_ret / abs(max_dd) if max_dd != 0 else 0

    var5  = np.percentile(r, 5)
    cvar5 = r[r <= var5].mean()

    return {
        "label":      label,
        "total_ret":  total,
        "ann_ret":    ann_ret,
        "ann_vol":    ann_vol,
        "sharpe":     sharpe,
        "sortino":    sortino,
        "max_dd":     max_dd,
        "calmar":     calmar,
        "win_rate":   (r > 0).mean(),
        "var_5pct":   var5,
        "cvar_5pct":  cvar5,
        "n_days":     n_days,
        "cum_returns": cum,
    }


# ── Strategies ────────────────────────────────────────────────────────────────

def run_backtest(
    btc_returns: pd.Series,
    predicted_log_rv: pd.Series,
    target_vol: float = 0.02,      # 2% daily target vol
    max_position: float = 1.5,     # max 1.5x BTC
    min_position: float = 0.1,     # always hold at least 10%
    tx_cost: float = 0.001,        # 0.1% per trade
) -> dict[str, dict]:
    """
    Run all strategies on aligned (btc_returns, predicted_log_rv) series.
    Returns dict of metric dicts keyed by strategy name.
    """
    # Align
    idx = btc_returns.index.intersection(predicted_log_rv.index)
    ret = btc_returns.loc[idx]
    pred_rv = np.exp(predicted_log_rv.loc[idx]) / np.sqrt(365)  # daily vol

    # ── Strategy 1: Buy & Hold ────────────────────────────────────────────────
    bh = ret.copy()

    # ── Strategy 2: Inverse-Vol Sizing ───────────────────────────────────────
    # Position today = target_daily_vol / predicted_daily_vol (from yesterday)
    position_iv = (target_vol / pred_rv).clip(min_position, max_position)
    position_iv = position_iv.shift(1).fillna(1.0)  # use yesterday's prediction

    trades_iv   = position_iv.diff().abs().fillna(0)
    cost_iv     = trades_iv * tx_cost
    ivol_ret    = position_iv * ret - cost_iv

    # ── Strategy 3: Vol Regime Filter ────────────────────────────────────────
    # Low vol (<33rd pct)  → 1.3x  |  Normal → 1.0x  |  High (>67th) → 0.5x
    q33 = pred_rv.quantile(0.33)
    q67 = pred_rv.quantile(0.67)

    position_rf = pd.Series(1.0, index=idx)
    position_rf[pred_rv < q33] = 1.3
    position_rf[pred_rv > q67] = 0.5
    position_rf = position_rf.clip(min_position, max_position)
    position_rf = position_rf.shift(1).fillna(1.0)

    trades_rf   = position_rf.diff().abs().fillna(0)
    cost_rf     = trades_rf * tx_cost
    regime_ret  = position_rf * ret - cost_rf

    # ── Strategy 4: Combined ─────────────────────────────────────────────────
    position_combo = ((target_vol / pred_rv) * position_rf).clip(min_position, max_position)
    position_combo = position_combo.shift(1).fillna(1.0)

    trades_combo  = position_combo.diff().abs().fillna(0)
    cost_combo    = trades_combo * tx_cost
    combo_ret     = position_combo * ret - cost_combo

    results = {
        "Buy & Hold":          compute_metrics(bh,         "Buy & Hold"),
        "Inverse-Vol Sizing":  compute_metrics(ivol_ret,   "Inverse-Vol Sizing"),
        "Vol Regime Filter":   compute_metrics(regime_ret, "Vol Regime Filter"),
        "Combined Strategy":   compute_metrics(combo_ret,  "Combined Strategy"),
    }
    return results, position_iv, position_rf, position_combo


# ── Main ──────────────────────────────────────────────────────────────────────

def run(model_name: str = "HAR-LGBM Hybrid"):

    # Load BTC daily returns
    btc_1d = pd.read_parquet(RAW_DIR / "BTCUSDT_1d.parquet")
    btc_1d["timestamp"] = pd.to_datetime(btc_1d["timestamp"], utc=True)
    btc_1d = btc_1d.set_index("timestamp").sort_index()
    btc_returns = np.log(btc_1d["close"] / btc_1d["close"].shift(1)).dropna()
    btc_returns.name = "btc_ret"

    # Load model predictions
    fname_map = {
        "HAR-RV-SJ":       "preds_har_rv_sj_h1.parquet",
        "HAR-Lasso":       "preds_har_lasso_h1.parquet",
        "HAR-LGBM Hybrid": "preds_har_lgbm_hybrid_h1.parquet",
        "HAR-LSTM Hybrid": "preds_har_lstm_hybrid_h1.parquet",
    }

    available = {k: v for k, v in fname_map.items() if (RESULTS / v).exists()}
    print(f"Models available for backtest: {list(available.keys())}")

    all_strategy_results = {}

    for mname, fname in available.items():
        preds = pd.read_parquet(RESULTS / fname).iloc[:-1]
        predicted_log_rv = preds["predicted"]

        strat_results, pos_iv, pos_rf, pos_combo = run_backtest(btc_returns, predicted_log_rv)
        all_strategy_results[mname] = strat_results

    # ── Print summary for best model ─────────────────────────────────────────
    best = all_strategy_results.get(model_name) or list(all_strategy_results.values())[0]
    best_name = model_name if model_name in all_strategy_results else list(all_strategy_results.keys())[0]

    print(f"\n{'='*72}")
    print(f"BACKTEST RESULTS — {best_name} predictions")
    print(f"{'='*72}")
    print(f"{'Strategy':<24} {'Total Ret':>10} {'Ann Ret':>9} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Max DD':>9} {'Calmar':>8} {'Win%':>7}")
    print("-"*80)
    for sname, m in best.items():
        print(f"{sname:<24} {m['total_ret']:>9.1%} {m['ann_ret']:>8.1%} "
              f"{m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['max_dd']:>8.1%} "
              f"{m['calmar']:>8.2f} {m['win_rate']:>6.1%}")

    print(f"\n{'='*50}")
    print("RISK METRICS (daily)")
    print(f"{'='*50}")
    print(f"{'Strategy':<24} {'VaR 5%':>9} {'CVaR 5%':>10} {'Ann Vol':>9}")
    print("-"*54)
    for sname, m in best.items():
        print(f"{sname:<24} {m['var_5pct']:>8.2%} {m['cvar_5pct']:>9.2%} {m['ann_vol']:>8.1%}")

    # ── Cross-model Sharpe comparison ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SHARPE RATIO BY MODEL & STRATEGY")
    print(f"{'='*60}")
    strats = ["Buy & Hold", "Inverse-Vol Sizing", "Vol Regime Filter", "Combined Strategy"]
    print(f"{'Model':<22}", end="")
    for s in strats: print(f"  {s[:18]:>18}", end="")
    print()
    print("-"*100)
    for mname, strat_dict in all_strategy_results.items():
        print(f"{mname:<22}", end="")
        for s in strats:
            print(f"  {strat_dict[s]['sharpe']:>18.3f}", end="")
        print()

    # ── Charts ────────────────────────────────────────────────────────────────
    print("\nGenerating backtest charts...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(f"Bitcoin Volatility Model Backtest — {best_name}", fontsize=13, fontweight='bold')

    colors = {
        "Buy & Hold":         "#95a5a6",
        "Inverse-Vol Sizing": "#e74c3c",
        "Vol Regime Filter":  "#3498db",
        "Combined Strategy":  "#2ecc71",
    }

    # Panel 1: Cumulative returns
    ax = axes[0]
    for sname, m in best.items():
        cum = m["cum_returns"]
        ax.plot(cum.index, cum.values, label=f"{sname}  (Sharpe {m['sharpe']:.2f})",
                color=colors[sname], linewidth=1.4 if sname != "Buy & Hold" else 0.9,
                alpha=0.9)
    ax.set_ylabel("Portfolio Value (start = 1.0)")
    ax.set_title("Cumulative Returns")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, fontsize=8)

    # Panel 2: Drawdown
    ax2 = axes[1]
    for sname, m in best.items():
        cum = m["cum_returns"]
        dd  = (cum - cum.cummax()) / cum.cummax()
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.35, color=colors[sname], label=sname)
    ax2.set_ylabel("Drawdown")
    ax2.set_title("Drawdown Over Time")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=25, fontsize=8)

    # Panel 3: Rolling position size (Combined Strategy)
    preds_df = pd.read_parquet(RESULTS / fname_map[best_name]).iloc[:-1]
    pred_rv_daily = np.exp(preds_df["predicted"]) / np.sqrt(365)
    idx_common = btc_returns.index.intersection(pred_rv_daily.index)
    pred_rv_aligned = pred_rv_daily.loc[idx_common]

    ax3 = axes[2]
    ax3.plot(pred_rv_aligned.index,
             pred_rv_aligned.rolling(7).mean() * np.sqrt(365) * 100,
             color="#e74c3c", linewidth=0.9, label="Predicted vol (7d avg, annualized %)")
    ax3_r = ax3.twinx()
    ax3_r.plot(pos_combo.index, pos_combo.values,
               color="#2ecc71", linewidth=0.8, alpha=0.7, label="BTC position size")
    ax3_r.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_ylabel("Predicted Annualized Vol (%)", color="#e74c3c")
    ax3_r.set_ylabel("Position Size (1.0 = 100% BTC)", color="#2ecc71")
    ax3.set_title("Predicted Volatility vs Combined Strategy Position Size")
    ax3.grid(alpha=0.2)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=25, fontsize=8)
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1+lines2, labels1+labels2, fontsize=9)

    plt.tight_layout()
    out = RESULTS / "backtest_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out.name}")

    return all_strategy_results


if __name__ == "__main__":
    run()
