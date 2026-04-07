"""
Post-hoc analysis utilities.

  - SHAP feature importance (requires shap package)
  - Mincer-Zarnowitz forecast efficiency regression
  - Rolling OOS performance plots
  - Regime-conditional error breakdown
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


# ── Mincer-Zarnowitz ──────────────────────────────────────────────────────────

def mincer_zarnowitz(actuals: pd.Series, preds: pd.Series) -> dict:
    """
    MZ regression: actual = alpha + beta * predicted + epsilon
    H0: alpha=0, beta=1 (unbiased, efficient forecast)
    """
    import statsmodels.api as sm
    X = sm.add_constant(preds.values)
    res = sm.OLS(actuals.values, X).fit(cov_type="HC3")
    alpha, beta = res.params
    alpha_pval, beta_pval = res.pvalues

    # Joint F-test: alpha=0, beta=1
    hypotheses = "const = 0, x1 = 1"
    f_test = res.f_test(["const = 0", "x1 = 1"])

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha_pval": float(alpha_pval),
        "beta_pval": float(beta_pval),
        "joint_f_pval": float(f_test.pvalue),
        "r2": float(res.rsquared),
        "unbiased": float(f_test.pvalue) > 0.05,
    }


# ── Rolling OOS performance ───────────────────────────────────────────────────

def rolling_qlike(actuals: pd.Series, preds: pd.Series, window: int = 60) -> pd.Series:
    """Rolling QLIKE loss over `window` days."""
    rv_true = np.exp(actuals)
    sigma2 = np.exp(preds)
    ratio = rv_true / sigma2
    loss = ratio - np.log(ratio) - 1
    return loss.rolling(window).mean()


def plot_rolling_performance(
    results_dict: dict[str, tuple[pd.Series, pd.Series]],
    window: int = 60,
    save_path: str | None = None,
) -> None:
    """
    Plot rolling QLIKE for multiple models.
    results_dict: {model_name: (actuals, predictions)}
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    for name, (actual, pred) in results_dict.items():
        rl = rolling_qlike(actual, pred, window=window)
        ax.plot(rl.index, rl.values, label=name, linewidth=1.2)

    ax.set_title(f"Rolling {window}-day QLIKE Loss (lower = better)", fontsize=13)
    ax.set_ylabel("QLIKE")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


# ── Regime Breakdown ──────────────────────────────────────────────────────────

def regime_breakdown(
    actuals: pd.Series,
    preds: pd.Series,
    rv_percentile: pd.Series,
) -> pd.DataFrame:
    """
    Break down QLIKE loss by volatility regime (low / mid / high).
    """
    from models import qlike_loss, mse_log

    regime = pd.cut(rv_percentile, bins=[0, 0.33, 0.67, 1.0],
                    labels=["low_vol", "mid_vol", "high_vol"])
    rows = []
    for label in ["low_vol", "mid_vol", "high_vol"]:
        mask = regime == label
        if mask.sum() == 0:
            continue
        a = actuals[mask].values
        p = preds[mask].values
        rows.append({
            "regime": label,
            "n_obs": int(mask.sum()),
            "qlike": qlike_loss(a, p),
            "mse_log": mse_log(a, p),
        })
    return pd.DataFrame(rows).set_index("regime")


# ── Actual vs Predicted plot ──────────────────────────────────────────────────

def plot_actual_vs_predicted(
    actuals: pd.Series,
    preds: pd.Series,
    model_name: str = "",
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series
    ax = axes[0]
    ax.plot(actuals.index, np.exp(actuals), label="Realized RV", alpha=0.7, linewidth=0.8)
    ax.plot(preds.index, np.exp(preds), label="Predicted RV", alpha=0.7, linewidth=0.8)
    ax.set_title(f"{model_name} — RV over time")
    ax.set_ylabel("Annualized RV")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

    # Scatter
    ax2 = axes[1]
    ax2.scatter(np.exp(preds), np.exp(actuals), alpha=0.3, s=8)
    lims = [0, max(np.exp(actuals).max(), np.exp(preds).max())]
    ax2.plot(lims, lims, "r--", linewidth=1, label="45° line")
    ax2.set_xlabel("Predicted RV")
    ax2.set_ylabel("Realized RV")
    ax2.set_title("Predicted vs Actual")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


# ── SHAP Feature Importance ───────────────────────────────────────────────────

def shap_importance(model, X_sample: pd.DataFrame, max_display: int = 20) -> pd.DataFrame:
    """
    Compute SHAP feature importance for tree-based models (XGBoost/LightGBM).
    Requires: pip install shap
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Run: pip install shap")

    # Works for XGBoost and LightGBM
    explainer = shap.TreeExplainer(model.model_)
    shap_vals = explainer.shap_values(X_sample.fillna(0))
    mean_abs = np.abs(shap_vals).mean(axis=0)

    df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(max_display)

    return df


if __name__ == "__main__":
    # Quick demo: load saved predictions and run MZ test
    import glob

    pred_files = list(RESULTS_DIR.glob("preds_*_h1.parquet"))
    if not pred_files:
        print("No prediction files found. Run pipeline.py first.")
    else:
        for f in pred_files:
            df = pd.read_parquet(f)
            mz = mincer_zarnowitz(df["actual"], df["predicted"])
            model_name = f.stem.replace("preds_", "").replace("_h1", "")
            print(f"\n{model_name}")
            print(f"  alpha={mz['alpha']:.4f} (p={mz['alpha_pval']:.3f}), "
                  f"beta={mz['beta']:.4f} (p={mz['beta_pval']:.3f}), "
                  f"R²={mz['r2']:.3f}, unbiased={mz['unbiased']}")
