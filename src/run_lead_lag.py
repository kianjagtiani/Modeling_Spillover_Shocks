"""
Run the lead-lag analysis: which assets Granger-cause BTC volatility?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from lead_lag import run_lead_lag_analysis
from config import PROCESSED_DIR

if __name__ == "__main__":
    # Use broad features if available (more assets), else base features
    broad_path = PROCESSED_DIR / "features_broad.parquet"
    base_path  = PROCESSED_DIR / "features.parquet"

    path = broad_path if broad_path.exists() else base_path
    print(f"Using: {path.name}")
    run_lead_lag_analysis(data_path=path)
