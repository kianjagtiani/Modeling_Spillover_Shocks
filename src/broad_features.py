"""
Feature engineering for the expanded 25+ asset universe.

Builds on the base features in features.py. All features are lagged by 1 day
(no lookahead). Handles missing assets gracefully — skips and warns if a
parquet file doesn't exist yet (run data_ingestion.py first).

New asset groups covered:
  Equity: NDX, DJI, MSCI World, XLK, XLF, SOXX
  Commodity: WTI, Brent, Copper
  Crypto (1d): ADA, DOT, AVAX, UNI, AAVE, MKR
  Composites: cross-market stress / spillover indices
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# New equity sector columns in traditional_assets.parquet
EQUITY_ASSETS = ["ndx", "dji", "msci_world", "xlk", "xlf", "soxx"]
COMMODITY_ASSETS = ["wti", "brent", "copper"]

# New crypto symbols to load from Binance 1d parquets
NEW_CRYPTO_SYMBOLS = {
    "ADAUSDT":  "ada",
    "DOTUSDT":  "dot",
    "AVAXUSDT": "avax",
    "UNIUSDT":  "uni",
    "AAVEUSDT": "aave",
    "MKRUSDT":  "mkr",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _log_ret(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))


def _realized_vol_5d(ret: pd.Series) -> pd.Series:
    return ret.rolling(5).std() * np.sqrt(252)


def _realized_vol_22d(ret: pd.Series) -> pd.Series:
    return ret.rolling(22).std() * np.sqrt(252)


# ── 1. Equity Sector Features ─────────────────────────────────────────────────

def equity_sector_features(trad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for NDX, DJI, MSCI World, XLK, XLF, SOXX.
    All lagged by 1 day. Forward-filled for weekends/holidays.
    """
    out = pd.DataFrame(index=trad_df.index)
    available = [a for a in EQUITY_ASSETS if a in trad_df.columns]

    if not available:
        _log("  [broad_features] No equity sector columns found in trad_df; skipping")
        return out

    for name in available:
        col = trad_df[name]
        ret = _log_ret(col)
        v5  = _realized_vol_5d(ret)
        v22 = _realized_vol_22d(ret)

        out[f"{name}_ret_1d"]  = ret.shift(1)
        out[f"{name}_vol_5d"]  = v5.shift(1)
        out[f"{name}_vol_22d"] = v22.shift(1)
        # Vol-regime: ratio of short to long vol (>1 = vol expanding, <1 = compressing)
        out[f"{name}_vol_ratio"] = (v5 / v22.replace(0, np.nan)).shift(1)

    # 52-week high proximity for each equity index (momentum signal)
    for name in available:
        col = trad_df[name]
        high_252 = col.rolling(252).max()
        out[f"{name}_52wk_proximity"] = (col / high_252.replace(0, np.nan)).shift(1)

    # Cross-sector relative signals
    if "xlk" in available and "xlf" in available:
        xlk_ret = _log_ret(trad_df["xlk"])
        xlf_ret = _log_ret(trad_df["xlf"])
        # Tech vs. financials: high ratio = risk appetite in tech/growth (positive for BTC vol)
        out["xlk_xlf_ratio"] = (xlk_ret / xlf_ret.replace(0, np.nan)).shift(1).clip(-5, 5)

    if "soxx" in available:
        soxx_ret = _log_ret(trad_df["soxx"])
        out["soxx_momentum_5d"] = soxx_ret.rolling(5).mean().shift(1)

    # Equity vol composite: average 5d vol across all available equity names
    vol_cols = [f"{n}_vol_5d" for n in available if f"{n}_vol_5d" in out.columns]
    if vol_cols:
        out["equity_vol_composite"] = out[vol_cols].mean(axis=1)

    return out


# ── 2. Commodity Features ─────────────────────────────────────────────────────

def commodity_features(trad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for WTI crude, Brent crude, and Copper.
    Commodity signals primarily act as global macro / risk-off indicators.
    """
    out = pd.DataFrame(index=trad_df.index)
    available = [a for a in COMMODITY_ASSETS if a in trad_df.columns]

    if not available:
        _log("  [broad_features] No commodity columns found in trad_df; skipping")
        return out

    for name in available:
        col = trad_df[name]
        ret = _log_ret(col)
        v5  = _realized_vol_5d(ret)

        out[f"{name}_ret_1d"] = ret.shift(1)
        out[f"{name}_vol_5d"] = v5.shift(1)

    # WTI–Brent spread (economic stress signal; large spread = market stress)
    if "wti" in available and "brent" in available:
        out["wti_brent_spread"] = np.log(
            trad_df["wti"] / trad_df["brent"].replace(0, np.nan)
        ).shift(1)

    # Oil vol spike: wti_vol_5d > 2× wti_vol_22d
    if "wti" in available:
        wti_ret = _log_ret(trad_df["wti"])
        v22 = _realized_vol_22d(wti_ret)
        v5  = _realized_vol_5d(wti_ret)
        out["oil_vol_spike"] = ((v5 > 2 * v22.replace(0, np.nan)).astype(int)).shift(1)

    # Copper 5d return — leading macro growth indicator (Dr. Copper)
    if "copper" in available:
        out["copper_momentum_5d"] = _log_ret(trad_df["copper"]).rolling(5).mean().shift(1)

    # Commodity risk index: average of oil and copper 1d returns (lagged)
    comm_ret_cols = [f"{n}_ret_1d" for n in available if f"{n}_ret_1d" in out.columns]
    if comm_ret_cols:
        out["commodity_risk_index"] = out[comm_ret_cols].mean(axis=1)

    return out


# ── 3. DeFi / Alt-Layer 1 Crypto Features ────────────────────────────────────

def defi_crypto_features(base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Load new crypto daily parquets (ADA, DOT, AVAX, UNI, AAVE, MKR)
    and build features. Skips assets whose parquet files don't exist.
    """
    out = pd.DataFrame(index=base_index)
    loaded = {}

    for symbol, name in NEW_CRYPTO_SYMBOLS.items():
        path = RAW_DIR / f"{symbol}_1d.parquet"
        if not path.exists():
            _log(f"  [broad_features] {symbol}_1d.parquet not found — run data_ingestion.py first")
            continue
        try:
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp").sort_index()
            loaded[name] = df
        except Exception as e:
            _log(f"  [broad_features] Failed to load {symbol}: {e}")

    if not loaded:
        return out

    for name, df in loaded.items():
        ret = _log_ret(df["close"])
        v5  = ret.rolling(5).std() * np.sqrt(365)  # crypto: 365 days

        ret_aligned = ret.reindex(base_index).ffill()
        v5_aligned  = v5.reindex(base_index).ffill()

        out[f"{name}_ret_1d"] = ret_aligned.shift(1)
        out[f"{name}_vol_5d"] = v5_aligned.shift(1)

    # Layer-1 altcoin vol composite (ADA, DOT, AVAX)
    l1_names  = [n for n in ["ada", "dot", "avax"] if f"{n}_vol_5d" in out.columns]
    l1_vol_cols = [f"{n}_vol_5d" for n in l1_names]
    if l1_vol_cols:
        out["altcoin_l1_vol_composite"] = out[l1_vol_cols].mean(axis=1)

    # DeFi vol composite (UNI, AAVE, MKR)
    defi_names  = [n for n in ["uni", "aave", "mkr"] if f"{n}_vol_5d" in out.columns]
    defi_vol_cols = [f"{n}_vol_5d" for n in defi_names]
    if defi_vol_cols:
        out["defi_vol_composite"] = out[defi_vol_cols].mean(axis=1)

    # DeFi momentum (average 5d return of DeFi tokens)
    defi_ret_cols = [f"{n}_ret_1d" for n in defi_names if f"{n}_ret_1d" in out.columns]
    if defi_ret_cols:
        out["defi_momentum"] = out[defi_ret_cols].mean(axis=1)

    return out


# ── 4. Cross-Market Stress Features ──────────────────────────────────────────

def cross_market_stress_features(
    trad_df: pd.DataFrame,
    base_index: pd.DatetimeIndex,
    base_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Composite cross-market stress and spillover indicators.
    Requires: trad_df with vix, sp500, plus any new equity/commodity columns.
    base_features_df: existing features.parquet for BTC vol context.
    """
    out = pd.DataFrame(index=base_index)

    # Global equity fear: composite of VIX + equity vol
    if "vix" in trad_df.columns:
        vix_norm = (trad_df["vix"] / 80).clip(0, 1)
        out["global_equity_fear"] = vix_norm.reindex(base_index).ffill().shift(1)

    # Risk-off signal: negative when markets are risk-off
    # Combines: S&P 500 return (neg = bad), DXY return (pos = dollar strengthening = risk-off)
    if "sp500" in trad_df.columns:
        sp_ret  = _log_ret(trad_df["sp500"])
        risk_off = -sp_ret  # negative SP return = risk-off
        if "dxy" in trad_df.columns:
            dxy_ret  = _log_ret(trad_df["dxy"])
            risk_off = risk_off + dxy_ret  # positive DXY = more risk-off
        out["risk_off_signal"] = risk_off.reindex(base_index).ffill().shift(1)

    # Equity-crypto vol ratio (when equity vol >> normal, crypto vol follows)
    if base_features_df is not None and "sp500_vol_5d" in base_features_df.columns:
        sp_vol = base_features_df["sp500_vol_5d"].reindex(base_index)
        if "rv_5d" in base_features_df.columns:
            btc_vol = base_features_df["rv_5d"].apply(np.exp).reindex(base_index)
            out["equity_crypto_vol_ratio"] = (sp_vol / btc_vol.replace(0, np.nan)).clip(0, 10)

    # Multi-asset drawdown: count of assets below 20-day moving average
    drawdown_count = pd.Series(0, index=trad_df.index)
    for col in ["sp500", "ndx", "xlk", "xlf", "gold"]:
        if col in trad_df.columns:
            ma20 = trad_df[col].rolling(20).mean()
            drawdown_count += (trad_df[col] < ma20).astype(int)
    if drawdown_count.sum() > 0:
        out["multi_asset_drawdown"] = drawdown_count.reindex(base_index).ffill().shift(1)

    # Cross-asset fear transmission: did VIX or equity vol spike yesterday?
    if "vix" in trad_df.columns:
        vix_spike = (trad_df["vix"] > trad_df["vix"].rolling(20).mean() * 1.3).astype(int)
        out["traditional_to_crypto_spillover_lag1"] = vix_spike.reindex(base_index).ffill().shift(1)

    return out


# ── 5. Master Builder ─────────────────────────────────────────────────────────

def build_broad_feature_matrix(features_df_path: Path | None = None) -> pd.DataFrame:
    """
    Load the existing features.parquet, join all new asset features, and save
    to data/processed/features_broad.parquet.

    Handles missing assets gracefully — skips any that haven't been downloaded.
    """
    # Load existing feature matrix
    base_path = features_df_path or (PROCESSED_DIR / "features.parquet")
    if not base_path.exists():
        raise FileNotFoundError(f"Base feature matrix not found: {base_path}\n"
                                f"Run: python src/pipeline.py --skip-ingest first.")

    _log(f"Loading base features from {base_path.name}...")
    df = pd.read_parquet(base_path)
    df.index = pd.to_datetime(df.index, utc=True)
    n_base = len(df.columns)
    _log(f"  Base: {len(df):,} rows × {n_base} cols ({df.index.min().date()} → {df.index.max().date()})")

    # Load traditional assets (expanded)
    trad_path = RAW_DIR / "traditional_assets.parquet"
    if not trad_path.exists():
        _log("  WARNING: traditional_assets.parquet not found — skipping equity/commodity features")
        trad_df = pd.DataFrame()
    else:
        trad_df = pd.read_parquet(trad_path)
        trad_df.index = pd.to_datetime(trad_df.index, utc=True)
        trad_df = trad_df.reindex(df.index).ffill()

    # Build and join all feature groups
    frames_to_join = []

    if not trad_df.empty:
        eq_feats = equity_sector_features(trad_df)
        if not eq_feats.empty:
            frames_to_join.append(eq_feats)
            _log(f"  + equity sector features: {eq_feats.shape[1]} cols")

        comm_feats = commodity_features(trad_df)
        if not comm_feats.empty:
            frames_to_join.append(comm_feats)
            _log(f"  + commodity features: {comm_feats.shape[1]} cols")

        stress_feats = cross_market_stress_features(trad_df, df.index, df)
        if not stress_feats.empty:
            frames_to_join.append(stress_feats)
            _log(f"  + cross-market stress features: {stress_feats.shape[1]} cols")

    defi_feats = defi_crypto_features(df.index)
    if not defi_feats.empty:
        frames_to_join.append(defi_feats)
        _log(f"  + DeFi/alt-L1 crypto features: {defi_feats.shape[1]} cols")

    # Join everything
    for feat_df in frames_to_join:
        feat_df.index = pd.to_datetime(feat_df.index, utc=True)
        df = df.join(feat_df, how="left")

    n_final = len(df.columns)
    _log(f"\nExpanded feature matrix: {n_base} → {n_final} cols (+{n_final - n_base})")
    _log(f"Null rate: {df.isnull().mean().mean():.1%}")

    out_path = PROCESSED_DIR / "features_broad.parquet"
    df.to_parquet(out_path)
    _log(f"Saved → {out_path}")
    return df


if __name__ == "__main__":
    build_broad_feature_matrix()
