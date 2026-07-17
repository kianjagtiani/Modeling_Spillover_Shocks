"""
Build the expanded feature matrix (broad assets: NDX, DJI, XLK, XLF, SOXX,
WTI, Brent, Copper, ADA, DOT, AVAX, UNI, AAVE, MKR) and add deep seasonality.

Prerequisites:
  1. python src/data_ingestion.py  (fetches all new tickers)
  2. python src/pipeline.py --skip-ingest  (builds base features.parquet)

Then run this script to produce:
  data/processed/features_broad.parquet       (base + new asset features)
  data/processed/features_seasonality.parquet (+ deep seasonality)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from broad_features import build_broad_feature_matrix
from seasonality_deep import add_seasonality_to_features

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1: Build broad asset feature matrix")
    print("=" * 60)
    df_broad = build_broad_feature_matrix()

    print()
    print("=" * 60)
    print("Step 2: Add deep seasonality features")
    print("=" * 60)
    from config import PROCESSED_DIR
    df_final = add_seasonality_to_features(PROCESSED_DIR / "features_broad.parquet")

    print()
    print(f"Final feature matrix: {df_final.shape[0]:,} rows × {df_final.shape[1]} cols")
    print("Ready to run: python src/pipeline.py --features-path data/processed/features_seasonality.parquet")
