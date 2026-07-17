"""
Central configuration for the BTC volatility forecasting project.
All magic numbers, asset lists, and path definitions live here.
Import this module instead of hard-coding constants across modules.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"

# Auto-create output dirs
for _d in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Data Sources ──────────────────────────────────────────────────────────────

BINANCE_BASE  = "https://api.binance.us/api/v3"
BINANCE_LIMIT = 1000   # max klines per request
SAVE_EVERY    = 50     # incremental save frequency

START_DATE = "2021-01-01"

# Core Binance symbols — fetched at 1d + 5m
CRYPTO_SYMBOLS_5M = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# Extended Binance symbols — fetched at 1d only
CRYPTO_SYMBOLS_1D = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "ADAUSDT", "DOTUSDT", "AVAXUSDT", "UNIUSDT", "AAVEUSDT", "MKRUSDT",
]

# yfinance tickers → column names in traditional_assets.parquet
TRAD_SYMBOLS = {
    "^VIX":     "vix",
    "^GSPC":    "sp500",
    "GLD":      "gold",
    "DX-Y.NYB": "dxy",
    "^TNX":     "tnx",
    "^NDX":     "ndx",
    "^DJI":     "dji",
    "URTH":     "msci_world",
    "XLK":      "xlk",
    "XLF":      "xlf",
    "SOXX":     "soxx",
    "CL=F":     "wti",
    "BZ=F":     "brent",
    "HG=F":     "copper",
}

# Grouping for feature engineering
EQUITY_INDICES  = ["sp500", "ndx", "dji", "msci_world"]
EQUITY_SECTORS  = ["xlk", "xlf", "soxx"]
COMMODITIES     = ["gold", "wti", "brent", "copper"]
CRYPTO_L1       = ["eth", "sol", "ada", "dot", "avax"]
CRYPTO_DEFI     = ["uni", "aave", "mkr"]


# ── Feature Engineering ───────────────────────────────────────────────────────

# HAR lag windows (days)
HAR_WINDOWS = [1, 5, 22]
HAR_COLS    = ["rv_1d", "rv_5d", "rv_22d", "jump_pos", "jump_neg"]

# Annualization
CRYPTO_ANNUALIZATION   = np.sqrt(365)   # 24/7 markets
EQUITY_ANNUALIZATION   = np.sqrt(252)   # traditional markets

# Realized vol floor (avoid log(0))
RV_LOWER_CLIP = 1e-8

# Rolling window for RV percentile regime signal
RV_PERCENTILE_WINDOW = 60

# Vol-of-vol windows
VOV_WINDOWS = [5, 22]

# Cross-asset correlation window
ALTCOIN_CORR_WINDOW = 10

# Bitcoin halving dates (UTC)
HALVING_DATES = pd.to_datetime([
    "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"
], utc=True)
NEXT_HALVING = pd.Timestamp("2028-04-14", tz="UTC")

# VIX spike threshold
VIX_SPIKE_THRESHOLD = 30


# ── Validation ────────────────────────────────────────────────────────────────

TRAIN_WINDOW_DAYS = 365
EMBARGO_DAYS      = 5
TEST_SIZE         = 1      # 1-day-ahead forecast
REFIT_EVERY       = 20     # refit model every N walk-forward steps
MIN_TRAIN_DAYS    = 365
VAL_FRACTION      = 0.20   # fraction of train used for early-stopping validation

# Forecast horizons to support
HORIZONS = [1, 5, 22]


# ── Models ────────────────────────────────────────────────────────────────────

# Default models to run in pipeline
DEFAULT_MODELS = [
    "HAR-RV-SJ",
    "HAR-Lasso",
    "LightGBM",
    "HAR-LGBM Hybrid",
    "HAR-LGBM Tuned",
    "Stacking Ensemble",
]

# Model-to-predictions filename mapping
MODEL_PRED_FILES = {
    "HAR-RV-SJ":       "preds_har_rv_sj_h1.parquet",
    "HAR-Lasso":       "preds_har_lasso_h1.parquet",
    "LightGBM":        "preds_lightgbm_h1.parquet",
    "XGBoost":         "preds_xgboost_h1.parquet",
    "HAR-LGBM Hybrid": "preds_har_lgbm_hybrid_h1.parquet",
    "HAR-LGBM Tuned":  "preds_har_lgbm_tuned_h1.parquet",
    "Stacking Ensemble": "preds_stacking_ensemble_h1.parquet",
    "HAR-LSTM Hybrid": "preds_har_lstm_hybrid_h1.parquet",
}

# LightGBM defaults
LGBM_DEFAULTS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# XGBoost defaults
XGBOOST_DEFAULTS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

# LSTM defaults
LSTM_DEFAULTS = dict(
    seq_len=22,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    learning_rate=1e-3,
    max_epochs=80,
    patience=10,
    batch_size=32,
)

# Optuna tuning
OPTUNA_N_TRIALS  = 60
OPTUNA_TUNE_FRAC = 0.60   # fraction of data used for HPO


# ── Backtest ──────────────────────────────────────────────────────────────────

TARGET_DAILY_VOL = 0.02    # 2% daily target volatility
MAX_POSITION     = 1.50    # max 1.5x BTC position
MIN_POSITION     = 0.10    # always hold at least 10%
TX_COST          = 0.001   # 0.1% transaction cost per trade

# Vol regime quantile thresholds (annualized)
VOL_LOW_ANNUALIZED  = 0.30   # < 30%  → low vol regime
VOL_HIGH_ANNUALIZED = 0.60   # > 60%  → high vol regime

# Regime-based position scaling
REGIME_POSITIONS = {
    "Low":    1.30,
    "Normal": 1.00,
    "High":   0.50,
}

# Multi-level vol targeting thresholds
MULTI_LEVEL_VOL_THRESHOLDS = {
    "low":    (0.0,   0.30, 0.03),   # (ann_vol_lo, ann_vol_hi, target_daily_vol)
    "normal": (0.30,  0.60, 0.02),
    "high":   (0.60, 99.0,  0.01),
}


# ── Options Strategy ──────────────────────────────────────────────────────────

VOL_PREMIUM      = 0.15   # typical Deribit IV premium over realized (~15%)
OPTIONS_EXPIRY_DAYS = 7   # weekly options
OPTIONS_TX_COST  = 0.05   # 5% of option price (Deribit taker fee)
VOL_EDGE_THRESHOLD = 0.15 # minimum vol edge to trigger a trade
MAX_VEGA_USD     = 500    # max vega exposure per position (USD)


# ── Lead-Lag Analysis ────────────────────────────────────────────────────────

GRANGER_MAX_LAG     = 10
CCF_MAX_LAG         = 15
ROLLING_WINDOW_DAYS = 90
VAR_LAGS            = 5
SPILLOVER_HORIZON   = 10
