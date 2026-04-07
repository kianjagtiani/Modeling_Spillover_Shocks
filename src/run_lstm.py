"""
Run walk-forward evaluation for HAR-LSTM Hybrid only,
then print comparison against previously saved model results.
"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from features import build_feature_matrix
from lstm_model import HARLSTMHybrid
from models import METRICS, qlike_loss, mse_log, mae_log
from validation import sliding_window_splits, walk_forward_evaluate

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
RESULTS   = Path(__file__).parent.parent / "results"

# Load feature matrix (already built)
print("Loading feature matrix...")
df = pd.read_parquet(PROCESSED / "features.parquet")
df.index = pd.to_datetime(df.index, utc=True)
df = df.iloc[:-1]   # drop incomplete last day

target      = "log_rv"
feature_cols = [c for c in df.columns if c != target]
print(f"Dataset: {len(df)} days x {len(feature_cols)} features")
print(f"Device: ", end="")
import torch
print("MPS" if torch.backends.mps.is_available() else "CPU")

# Walk-forward splits (same as main pipeline)
splits = sliding_window_splits(df.index, train_window=365, test_size=1, embargo=5)
print(f"Splits: {len(splits)}")

# Run LSTM
print("\n=== HAR-LSTM Hybrid ===")
result = walk_forward_evaluate(
    df=df,
    target=target,
    feature_cols=feature_cols,
    model_factory=lambda: HARLSTMHybrid(
        seq_len=22, hidden_size=64, num_layers=2,
        dropout=0.2, lr=1e-3, max_epochs=80, patience=10, batch_size=32
    ),
    metric_fns=METRICS,
    splits=splits,
    verbose=True,
    refit_every=20,
)

# Save predictions
pred_df = pd.DataFrame({
    "actual":    result.actuals,
    "predicted": result.predictions,
    "error":     result.actuals - result.predictions,
})
pred_df.to_parquet(RESULTS / "preds_har_lstm_hybrid_h1.parquet")

# Compare against saved models
print("\n" + "="*70)
print("FINAL COMPARISON  (all models, OOS walk-forward)")
print("="*70)

saved = {
    "HAR-RV-SJ":       "preds_har_rv_sj_h1.parquet",
    "HAR-Lasso":       "preds_har_lasso_h1.parquet",
    "XGBoost":         "preds_xgboost_h1.parquet",
    "LightGBM":        "preds_lightgbm_h1.parquet",
    "HAR-LGBM Hybrid": "preds_har_lgbm_hybrid_h1.parquet",
}

rows = []
for name, fname in saved.items():
    p = RESULTS / fname
    if not p.exists(): continue
    d = pd.read_parquet(p).iloc[:-1]   # drop incomplete day
    common = d.index.intersection(result.predictions.index)
    a, pr = d.loc[common, "actual"].values, d.loc[common, "predicted"].values
    rows.append({"model": name,
                 "qlike": qlike_loss(a, pr),
                 "mse_log": mse_log(a, pr),
                 "mae_log": mae_log(a, pr),
                 "corr": np.corrcoef(a, pr)[0,1]})

# Add LSTM
a_l = result.actuals.values
p_l = result.predictions.values
rows.append({"model": "HAR-LSTM Hybrid ★",
             "qlike": qlike_loss(a_l, p_l),
             "mse_log": mse_log(a_l, p_l),
             "mae_log": mae_log(a_l, p_l),
             "corr": np.corrcoef(a_l, p_l)[0,1]})

tbl = pd.DataFrame(rows).set_index("model").sort_values("mse_log")
print(tbl.round(5).to_string())
print()
print(f"LSTM scores: {result.scores}")
