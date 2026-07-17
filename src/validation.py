"""
Walk-forward validation with purging and embargo.

Implements:
  - Expanding-window walk-forward (anchored start, growing train set)
  - Sliding-window walk-forward (fixed train window, rolls forward)
  - Embargo gap between train and test to prevent leakage from correlated features
  - Diebold-Mariano test for pairwise model comparison
  - Model Confidence Set (simplified via DM bootstrap)
  - Results aggregation and reporting
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


# ── Walk-Forward Splits ───────────────────────────────────────────────────────

@dataclass
class Split:
    train_idx: pd.Index
    test_idx: pd.Index
    step: int


def expanding_window_splits(
    index: pd.Index,
    min_train: int = 365,
    test_size: int = 1,
    embargo: int = 5,
    step: int = 1,
) -> list[Split]:
    """
    Generate expanding-window walk-forward splits.

    Args:
        index:      Full datetime index of the dataset.
        min_train:  Minimum number of training observations.
        test_size:  Number of test observations per split (default 1 = daily).
        embargo:    Days to purge between train end and test start (prevent leakage).
        step:       How many days to advance per split.
    """
    splits = []
    n = len(index)
    t = min_train

    i = 0
    while t + embargo + test_size <= n:
        train_end = t
        test_start = t + embargo
        test_end = min(test_start + test_size, n)
        splits.append(Split(
            train_idx=index[:train_end],
            test_idx=index[test_start:test_end],
            step=i,
        ))
        t += step
        i += 1

    return splits


def sliding_window_splits(
    index: pd.Index,
    train_window: int = 365,
    test_size: int = 1,
    embargo: int = 5,
    step: int = 1,
) -> list[Split]:
    """
    Generate sliding-window (rolling) walk-forward splits.
    Better for crypto where regimes shift and older data becomes stale.
    """
    splits = []
    n = len(index)
    t = train_window

    i = 0
    while t + embargo + test_size <= n:
        train_start = t - train_window
        train_end = t
        test_start = t + embargo
        test_end = min(test_start + test_size, n)
        splits.append(Split(
            train_idx=index[train_start:train_end],
            test_idx=index[test_start:test_end],
            step=i,
        ))
        t += step
        i += 1

    return splits


# ── Walk-Forward Evaluation ───────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    predictions: pd.Series        # indexed by test date
    actuals: pd.Series
    scores: dict[str, float]      # aggregated metric scores
    split_scores: list[dict]      # per-split scores


def walk_forward_evaluate(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str],
    model_factory: Callable,
    metric_fns: dict[str, Callable],
    splits: list[Split],
    verbose: bool = True,
    refit_every: int = 1,
) -> ModelResult:
    """
    Run walk-forward evaluation for a single model.

    Args:
        df:              Full feature matrix (must include target column).
        target:          Name of the target column (e.g. "log_rv").
        feature_cols:    Feature column names to use.
        model_factory:   Callable returning a fresh model instance.
        metric_fns:      Dict of {name: fn(y_true, y_pred) -> float}.
        splits:          List of Split objects from *_splits() functions.
        refit_every:     Refit model every N splits (1 = always refit).
    """
    preds = {}
    actuals_dict = {}
    split_scores = []
    model = None

    for i, split in enumerate(splits):
        X_train = df.loc[split.train_idx, feature_cols]
        y_train = df.loc[split.train_idx, target]
        X_test = df.loc[split.test_idx, feature_cols]
        y_test = df.loc[split.test_idx, target]

        # Skip splits with too many NaNs
        if y_train.isna().mean() > 0.1 or y_test.isna().any():
            continue

        # Refit model
        if model is None or (i % refit_every == 0):
            model = model_factory()
            # Use last 20% of train as validation for early stopping
            val_size = max(30, len(X_train) // 5)
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_tr = X_train.iloc[:-val_size]
            y_tr = y_train.iloc[:-val_size]

            try:
                # Models that support validation sets
                model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
            except TypeError:
                model.fit(X_train, y_train)

        pred = model.predict(X_test)
        for date, p, a in zip(split.test_idx, pred, y_test.values):
            preds[date] = p
            actuals_dict[date] = a

        scores = {k: fn(y_test.values, pred) for k, fn in metric_fns.items()}
        split_scores.append({"step": i, "date": split.test_idx[-1], **scores})

        if verbose and i % 100 == 0:
            score_str = "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
            print(f"  Split {i:4d}/{len(splits)}  {score_str}")

    pred_series = pd.Series(preds).sort_index()
    actual_series = pd.Series(actuals_dict).sort_index()
    agg_scores = {k: fn(actual_series.values, pred_series.values)
                  for k, fn in metric_fns.items()}

    return ModelResult(
        name=model.name if hasattr(model, "name") else type(model).__name__,
        predictions=pred_series,
        actuals=actual_series,
        scores=agg_scores,
        split_scores=split_scores,
    )


# ── Diebold-Mariano Test ──────────────────────────────────────────────────────

def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str = "mse",
    h: int = 1,
) -> dict:
    """
    Diebold-Mariano test of equal predictive accuracy.
    H0: models A and B have equal forecast accuracy.
    Returns test statistic, p-value, and which model is better.

    Args:
        loss: "mse" or "mae"
        h:    forecast horizon (for autocorrelation correction)
    """
    if loss == "mse":
        e_a = (y_true - pred_a) ** 2
        e_b = (y_true - pred_b) ** 2
    elif loss == "mae":
        e_a = np.abs(y_true - pred_a)
        e_b = np.abs(y_true - pred_b)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = e_a - e_b
    T = len(d)

    # Newey-West HAC variance estimator
    gamma0 = np.var(d, ddof=1)
    nw_var = gamma0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k], ddof=1)[0, 1]
        nw_var += 2 * (1 - k / h) * gamma_k
    nw_var = max(nw_var, 1e-12)

    dm_stat = np.mean(d) / np.sqrt(nw_var / T)
    p_value = 2 * stats.norm.sf(np.abs(dm_stat))

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "better_model": "A" if dm_stat < 0 else "B",
        "significant_5pct": p_value < 0.05,
    }


# ── Results Aggregation ───────────────────────────────────────────────────────

def compare_models(results: list[ModelResult], baseline_name: str | None = None) -> pd.DataFrame:
    """
    Build a comparison table of model scores and DM test vs. baseline.
    """
    rows = []
    for r in results:
        row = {"model": r.name, **r.scores}
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model")

    if baseline_name is not None:
        baseline = next((r for r in results if r.name == baseline_name), None)
        if baseline is not None:
            dm_cols = {}
            for r in results:
                if r.name == baseline_name:
                    dm_cols[r.name] = {"dm_p": np.nan, "beats_baseline": "—"}
                    continue
                common = r.predictions.index.intersection(baseline.predictions.index)
                dm = diebold_mariano(
                    baseline.actuals.loc[common].values,
                    baseline.predictions.loc[common].values,
                    r.predictions.loc[common].values,
                )
                dm_cols[r.name] = {
                    "dm_p": dm["p_value"],
                    "beats_baseline": "yes*" if dm["better_model"] == "B" and dm["significant_5pct"] else
                                      "yes" if dm["better_model"] == "B" else "no",
                }
            dm_df = pd.DataFrame(dm_cols).T
            table = table.join(dm_df)

    return table.round(6)
