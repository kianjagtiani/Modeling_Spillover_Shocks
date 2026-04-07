"""
Main pipeline entry point.

Runs the full workflow:
  1. Ingest raw data (Binance + Fear & Greed)
  2. Build feature matrix
  3. Walk-forward evaluation across all models
  4. Save results and print comparison table

Usage:
  python src/pipeline.py [--start 2020-01-01] [--window sliding|expanding]
                         [--horizon 1] [--train-days 365]
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from data_ingestion import ingest_all
from features import build_feature_matrix
from models import HARModel, HARLassoModel, XGBoostModel, LGBMModel, HARLGBMHybrid, METRICS
from optimize import HARLGBMTuned, StackingEnsemble
from lstm_model import HARLSTMHybrid
from validation import (
    expanding_window_splits,
    sliding_window_splits,
    walk_forward_evaluate,
    compare_models,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def get_feature_cols(df: pd.DataFrame, target: str = "log_rv") -> list[str]:
    """Return all columns except the target."""
    return [c for c in df.columns if c != target]


def run_pipeline(
    start: str = "2021-01-01",
    window: str = "sliding",
    horizon: int = 1,
    train_days: int = 365,
    embargo: int = 5,
    skip_ingest: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:

    # ── 1. Ingest ──────────────────────────────────────────────────────────────
    if not skip_ingest:
        ingest_all(start=start)
    else:
        print("Skipping ingestion (--skip-ingest)")

    # ── 2. Features ────────────────────────────────────────────────────────────
    features_path = PROCESSED_DIR / "features.parquet"
    if features_path.exists() and skip_ingest:
        print("Loading cached feature matrix...")
        df = pd.read_parquet(features_path)
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df = build_feature_matrix(start=start)

    target = "log_rv"
    feature_cols = get_feature_cols(df, target)

    if verbose:
        print(f"\nDataset: {len(df)} days × {len(feature_cols)} features")
        print(f"Target range: [{df[target].min():.3f}, {df[target].max():.3f}]")
        print(f"Validation window: {window}, horizon: h={horizon}, "
              f"train_days: {train_days}, embargo: {embargo}\n")

    # ── 3. Splits ──────────────────────────────────────────────────────────────
    if window == "expanding":
        splits = expanding_window_splits(
            df.index, min_train=train_days, test_size=horizon, embargo=embargo
        )
    else:
        splits = sliding_window_splits(
            df.index, train_window=train_days, test_size=horizon, embargo=embargo
        )

    if verbose:
        print(f"Total walk-forward splits: {len(splits)}\n")

    # ── 4. Models ──────────────────────────────────────────────────────────────
    tuned_model = HARLGBMTuned(n_trials=60)  # shared instance so tuning happens once

    base_factories = {
        "HAR-RV-SJ": lambda: HARModel(),
        "HAR-Lasso": lambda: HARLassoModel(),
        "LightGBM":  lambda: LGBMModel(),
        "HAR-LGBM Hybrid": lambda: HARLGBMHybrid(),
    }

    model_factories = {
        "HAR-RV-SJ":         lambda: HARModel(),
        "HAR-Lasso":         lambda: HARLassoModel(),
        "XGBoost":           lambda: XGBoostModel(),
        "LightGBM":          lambda: LGBMModel(),
        "HAR-LGBM Hybrid":   lambda: HARLGBMHybrid(),
        "HAR-LGBM Tuned":    lambda: tuned_model,
        "Stacking Ensemble": lambda: StackingEnsemble(base_factories),
        "HAR-LSTM Hybrid":   lambda: HARLSTMHybrid(seq_len=22, hidden_size=64,
                                                    num_layers=2, max_epochs=80,
                                                    patience=10),
    }

    all_results = []
    for model_name, factory in model_factories.items():
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")
        result = walk_forward_evaluate(
            df=df,
            target=target,
            feature_cols=feature_cols,
            model_factory=factory,
            metric_fns=METRICS,
            splits=splits,
            verbose=verbose,
            refit_every=20,   # refit every 20 steps (~monthly) for speed
        )
        all_results.append(result)
        score_str = "  ".join(f"{k}={v:.5f}" for k, v in result.scores.items())
        print(f"  FINAL → {score_str}\n")

    # ── 5. Compare ─────────────────────────────────────────────────────────────
    comparison = compare_models(all_results, baseline_name="HAR-RV-SJ")

    print("\n" + "="*70)
    print("MODEL COMPARISON (OOS walk-forward)")
    print("="*70)
    print(comparison.to_string())
    print()

    # ── 6. Save ────────────────────────────────────────────────────────────────
    comparison.to_csv(RESULTS_DIR / f"comparison_h{horizon}_{window}.csv")

    for r in all_results:
        pred_df = pd.DataFrame({
            "actual": r.actuals,
            "predicted": r.predictions,
            "error": r.actuals - r.predictions,
        })
        safe_name = r.name.replace(" ", "_").replace("-", "_").lower()
        pred_df.to_parquet(RESULTS_DIR / f"preds_{safe_name}_h{horizon}.parquet")

    # Save scores as JSON
    scores_out = {
        r.name: {**r.scores, "n_obs": len(r.predictions)}
        for r in all_results
    }
    with open(RESULTS_DIR / f"scores_h{horizon}_{window}.json", "w") as f:
        json.dump(scores_out, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}/")
    return comparison


def main():
    parser = argparse.ArgumentParser(description="BTC volatility forecasting pipeline")
    parser.add_argument("--start", default="2021-01-01", help="Data start date (YYYY-MM-DD)")
    parser.add_argument("--window", choices=["sliding", "expanding"], default="sliding")
    parser.add_argument("--horizon", type=int, default=1, help="Forecast horizon in days")
    parser.add_argument("--train-days", type=int, default=365, help="Training window size")
    parser.add_argument("--embargo", type=int, default=5, help="Embargo days between train/test")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data download")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    run_pipeline(
        start=args.start,
        window=args.window,
        horizon=args.horizon,
        train_days=args.train_days,
        embargo=args.embargo,
        skip_ingest=args.skip_ingest,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
