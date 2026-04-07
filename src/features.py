"""
Feature engineering pipeline.

Constructs:
  - Realized volatility (RV) from 5-min returns
  - Bipower variation and signed jumps (HAR-RV-SJ components)
  - HAR lags: daily / weekly / monthly
  - Vol-of-vol features
  - Cross-asset features (ETH, SOL volatility; altcoin correlation)
  - Seasonality features (cyclical encoding + event dummies)
  - Sentiment features (Fear & Greed)
  - Garman-Klass estimator (daily OHLCV fallback)
  - Regime features (rolling RV percentile)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Halving dates (UTC midnight) ─────────────────────────────────────────────
HALVING_DATES = pd.to_datetime([
    "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"
], utc=True)

# US market holidays (approximate; extend as needed)
US_HOLIDAYS = pd.to_datetime([
    "2019-01-01", "2019-07-04", "2019-11-28", "2019-12-25",
    "2020-01-01", "2020-07-04", "2020-11-26", "2020-12-25",
    "2021-01-01", "2021-07-05", "2021-11-25", "2021-12-24",
    "2022-01-17", "2022-02-21", "2022-05-30", "2022-07-04",
    "2022-09-05", "2022-11-24", "2022-12-26",
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-05-29",
    "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-05-27",
    "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26",
    "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
], utc=True).normalize()


# ── Realized Volatility ───────────────────────────────────────────────────────

def compute_rv(df_5m: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Compute daily realized volatility from 5-min OHLCV data.
    RV_t = sqrt( sum of squared 5-min log-returns )
    Annualized by sqrt(365 * 288) for 24/7 crypto markets.
    """
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df.loc[df["timestamp"].dt.floor("5min") == df["date"], "log_ret"] = np.nan  # drop first bar of day

    # rv_daily = sqrt(sum of squared 5-min log-returns) = daily realized vol
    rv = df.groupby("date")["log_ret"].apply(lambda r: np.sqrt((r**2).sum()))
    rv.name = "rv"

    # Annualize by sqrt(365) — rv is already a daily vol figure, not a per-bar figure
    if annualize:
        rv = rv * np.sqrt(365)

    return rv


def compute_bipower_variation(df_5m: pd.DataFrame) -> pd.Series:
    """
    Bipower variation (Barndorff-Nielsen & Shephard 2004) for jump separation.
    BV_t = (π/2) * sum(|r_i| * |r_{i-1}|)
    """
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["abs_ret"] = np.abs(np.log(df["close"] / df["close"].shift(1)))
    df["bv_contrib"] = df["abs_ret"] * df["abs_ret"].shift(1)

    # BV is in variance units (sum of products of abs returns), annualize as variance then sqrt
    bv = np.sqrt(df.groupby("date")["bv_contrib"].sum() * (np.pi / 2) * 365)
    bv.name = "bv"
    return bv


def compute_signed_jumps(rv: pd.Series, bv: pd.Series, df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Signed jump components (Patton & Sheppard 2015).
    Jump = max(RV - BV, 0)
    Positive/negative jumps split by sign of intraday returns during high-vol minutes.
    """
    df = df_5m.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    # Intraday return sign (dominant direction on jump days)
    sign = df.groupby("date")["log_ret"].sum().apply(np.sign)

    jump_raw = (rv**2 - bv).clip(lower=0)
    jump = np.sqrt(jump_raw)

    j_pos = (jump * (sign > 0)).rename("jump_pos")
    j_neg = (jump * (sign < 0)).rename("jump_neg")

    return pd.DataFrame({"jump_pos": j_pos, "jump_neg": j_neg})


# ── Garman-Klass (daily OHLCV fallback) ──────────────────────────────────────

def garman_klass(df_1d: pd.DataFrame, annualize: bool = True) -> pd.Series:
    """
    Garman-Klass volatility estimator from daily OHLCV.
    More efficient than close-to-close; valid for 24/7 continuous trading.
    """
    ln_hl = np.log(df_1d["high"] / df_1d["low"])
    ln_co = np.log(df_1d["close"] / df_1d["open"])
    gk = np.sqrt(0.5 * ln_hl**2 - (2 * np.log(2) - 1) * ln_co**2)
    if annualize:
        gk = gk * np.sqrt(365)
    gk.name = "rv_gk"
    return gk


# ── HAR Components ────────────────────────────────────────────────────────────

def har_components(log_rv: pd.Series) -> pd.DataFrame:
    """
    Build HAR lags of log-realized volatility.
    Daily (lag 1), weekly (5-day avg), monthly (22-day avg).
    """
    rv_d = log_rv.shift(1).rename("rv_1d")
    rv_w = log_rv.shift(1).rolling(5).mean().rename("rv_5d")
    rv_m = log_rv.shift(1).rolling(22).mean().rename("rv_22d")
    return pd.concat([rv_d, rv_w, rv_m], axis=1)


def vol_of_vol(log_rv: pd.Series) -> pd.DataFrame:
    """Rolling std of log(RV) — captures vol-of-vol regime."""
    vov_5 = log_rv.shift(1).rolling(5).std().rename("vov_5d")
    vov_22 = log_rv.shift(1).rolling(22).std().rename("vov_22d")
    return pd.concat([vov_5, vov_22], axis=1)


def rv_percentile(log_rv: pd.Series, window: int = 60) -> pd.Series:
    """
    Rolling percentile rank of current RV vs past `window` days.
    Soft regime signal: high percentile = elevated vol regime.
    """
    def _pct(x):
        return (x.iloc[:-1] < x.iloc[-1]).mean()

    return log_rv.shift(1).rolling(window + 1).apply(_pct, raw=False).rename("rv_pct_60d")


# ── Cross-Asset Features ──────────────────────────────────────────────────────

def cross_asset_features(daily_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build cross-asset features from dict of {symbol: df_1d}.
    Includes: log-returns, realized vol proxies, altcoin correlation.
    """
    frames = {}
    for sym, df in daily_data.items():
        if sym == "BTCUSDT":
            continue
        tag = sym.replace("USDT", "").lower()
        df = df.set_index("timestamp")
        ret = np.log(df["close"] / df["close"].shift(1))
        vol = ret.rolling(5).std() * np.sqrt(365)
        frames[f"{tag}_ret_1d"] = ret.shift(1)
        frames[f"{tag}_vol_5d"] = vol.shift(1)

    # Altcoin correlation: rolling 10d corr between BTC and ETH daily log-returns
    if "BTCUSDT" in daily_data and "ETHUSDT" in daily_data:
        btc_ret = np.log(
            daily_data["BTCUSDT"].set_index("timestamp")["close"] /
            daily_data["BTCUSDT"].set_index("timestamp")["close"].shift(1)
        )
        eth_ret = np.log(
            daily_data["ETHUSDT"].set_index("timestamp")["close"] /
            daily_data["ETHUSDT"].set_index("timestamp")["close"].shift(1)
        )
        frames["btc_eth_corr_10d"] = btc_ret.rolling(10).corr(eth_ret).shift(1)

    return pd.DataFrame(frames)


def traditional_asset_features(trad: pd.DataFrame) -> pd.DataFrame:
    """
    Features derived from traditional financial assets.
    All lagged by 1 day to avoid lookahead.

    VIX:   level, 5d change, spike dummy (>30), rolling 5d avg
    SP500: log-return, 5d realized vol, rolling corr with BTC
    Gold:  log-return, 5d vol
    DXY:   log-return, 5d vol (negative predictor of BTC vol)
    TNX:   yield level, 5d change (rate shock signal)
    """
    df = trad.copy().sort_index()
    out = pd.DataFrame(index=df.index)

    # VIX — most important traditional predictor
    if "vix" in df.columns:
        out["vix_level"]    = df["vix"].shift(1)
        out["vix_5d_avg"]   = df["vix"].shift(1).rolling(5).mean()
        out["vix_5d_chg"]   = df["vix"].diff(5).shift(1)
        out["vix_spike"]    = (df["vix"].shift(1) > 30).astype(int)
        out["log_vix"]      = np.log(df["vix"].clip(lower=1)).shift(1)

    # S&P 500
    if "sp500" in df.columns:
        sp_ret = np.log(df["sp500"] / df["sp500"].shift(1))
        out["sp500_ret_1d"]  = sp_ret.shift(1)
        out["sp500_vol_5d"]  = sp_ret.rolling(5).std().shift(1) * np.sqrt(252)
        out["sp500_vol_22d"] = sp_ret.rolling(22).std().shift(1) * np.sqrt(252)
        # Range-based vol (Parkinson) — available even on non-trading days
        if "sp500_high" in df.columns and "sp500_low" in df.columns:
            park = np.sqrt(np.log(df["sp500_high"] / df["sp500_low"]) ** 2 / (4 * np.log(2)))
            out["sp500_park_vol"] = park.shift(1)

    # Gold
    if "gold" in df.columns:
        gold_ret = np.log(df["gold"] / df["gold"].shift(1))
        out["gold_ret_1d"]  = gold_ret.shift(1)
        out["gold_vol_5d"]  = gold_ret.rolling(5).std().shift(1) * np.sqrt(252)

    # DXY — dollar strength (negative predictor)
    if "dxy" in df.columns:
        dxy_ret = np.log(df["dxy"] / df["dxy"].shift(1))
        out["dxy_ret_1d"]  = dxy_ret.shift(1)
        out["dxy_vol_5d"]  = dxy_ret.rolling(5).std().shift(1) * np.sqrt(252)

    # 10Y yield — rate shock signal
    if "tnx" in df.columns:
        out["tnx_level"]  = df["tnx"].shift(1)
        out["tnx_5d_chg"] = df["tnx"].diff(5).shift(1)

    return out


def volume_features(df_1d: pd.DataFrame) -> pd.DataFrame:
    """Relative volume and quote volume vs rolling baseline."""
    df = df_1d.set_index("timestamp").copy()
    vol_ma = df["volume"].rolling(7).mean()
    qvol_ma = df["quote_volume"].rolling(7).mean()
    return pd.DataFrame({
        "vol_ratio": (df["volume"] / vol_ma).shift(1),
        "qvol_ratio": (df["quote_volume"] / qvol_ma).shift(1),
    })


# ── Seasonality ───────────────────────────────────────────────────────────────

def seasonality_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Cyclical encoding of calendar features + event dummies.
    All continuous features use sin/cos to preserve boundary continuity.
    """
    dow = index.dayofweek        # 0=Mon … 6=Sun
    month = index.month          # 1–12
    doy = index.dayofyear        # 1–366

    feats = pd.DataFrame(index=index)

    # Cyclical: day-of-week
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Cyclical: month
    feats["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    feats["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Cyclical: day-of-year (annual seasonality)
    feats["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    feats["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # Weekend dummy (Sat/Sun → higher vol)
    feats["is_weekend"] = (dow >= 5).astype(int)

    # US holiday dummy (elevated BTC vol)
    date_only = index.normalize()
    feats["is_us_holiday"] = date_only.isin(US_HOLIDAYS).astype(int)

    # Halving cycle phase: continuous variable [0, 1] within each ~4-year cycle
    # Uses the most recent past halving as cycle start
    days_in_cycle = 1461  # ~4 years
    def _halving_phase(ts):
        past = HALVING_DATES[HALVING_DATES <= ts]
        if len(past) == 0:
            return 0.0
        last_halving = past[-1]
        days_since = (ts - last_halving).days
        return (days_since % days_in_cycle) / days_in_cycle

    feats["halving_phase_sin"] = np.sin(
        2 * np.pi * np.array([_halving_phase(ts) for ts in index])
    )
    feats["halving_phase_cos"] = np.cos(
        2 * np.pi * np.array([_halving_phase(ts) for ts in index])
    )

    # Quarter end (last 3 days of quarter → elevated vol)
    feats["is_quarter_end"] = (
        (index.is_quarter_end) |
        (pd.Series(index).apply(
            lambda t: (t + pd.Timedelta(days=1)).is_quarter_end or
                      (t + pd.Timedelta(days=2)).is_quarter_end
        ).values)
    ).astype(int)

    return feats


# ── Sentiment ─────────────────────────────────────────────────────────────────

def sentiment_features(fear_greed: pd.DataFrame) -> pd.DataFrame:
    """
    Fear & Greed index features.
    Extreme fear/greed predicts short-horizon vol persistence.
    """
    fg = fear_greed.set_index("timestamp")[["fear_greed"]].copy()
    fg.index = fg.index.normalize()
    fg["fg_extreme_fear"] = (fg["fear_greed"] < 20).astype(int)
    fg["fg_extreme_greed"] = (fg["fear_greed"] > 80).astype(int)
    fg["fg_lag1"] = fg["fear_greed"].shift(1)
    fg["fg_7d_chg"] = fg["fear_greed"].diff(7).shift(1)
    return fg


# ── Master Feature Builder ────────────────────────────────────────────────────

def build_feature_matrix(start: str = "2019-01-01") -> pd.DataFrame:
    """
    Assemble the full feature matrix and save to processed/.
    Returns a DataFrame with target `log_rv` and all features,
    indexed by UTC date, with no lookahead.
    """
    print("=== Building Feature Matrix ===")

    # Load raw data
    btc_5m = pd.read_parquet(RAW_DIR / "BTCUSDT_5m.parquet")
    btc_5m["timestamp"] = pd.to_datetime(btc_5m["timestamp"], utc=True)

    btc_1d = pd.read_parquet(RAW_DIR / "BTCUSDT_1d.parquet")
    btc_1d["timestamp"] = pd.to_datetime(btc_1d["timestamp"], utc=True)

    daily_data = {"BTCUSDT": btc_1d}
    for sym in ["ETHUSDT", "SOLUSDT", "BNBUSDT"]:
        path = RAW_DIR / f"{sym}_1d.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            daily_data[sym] = df

    # Target: realized volatility from 5-min data
    print("  Computing RV from 5-min data...")
    btc_5m["timestamp"] = pd.to_datetime(btc_5m["timestamp"], utc=True)
    rv = compute_rv(btc_5m)
    bv = compute_bipower_variation(btc_5m)

    rv.index = pd.to_datetime(rv.index, utc=True)
    bv.index = pd.to_datetime(bv.index, utc=True)

    # Signed jumps
    jumps = compute_signed_jumps(rv, bv, btc_5m)
    jumps.index = pd.to_datetime(jumps.index, utc=True)

    log_rv = np.log(rv.clip(lower=1e-8)).rename("log_rv")

    # HAR components
    har = har_components(log_rv)
    vov = vol_of_vol(log_rv)
    pct = rv_percentile(log_rv)

    # Realized semi-variances (upside / downside decomposition)
    print("  Computing realized semi-variances...")
    from optimize import realized_semi_variances
    semi_var = realized_semi_variances(btc_5m)
    semi_var.index = pd.to_datetime(semi_var.index, utc=True)

    # Garman-Klass (daily fallback, use as additional feature)
    gk = garman_klass(btc_1d.set_index("timestamp"))
    log_gk = np.log(gk.clip(lower=1e-8)).shift(1).rename("log_rv_gk_lag1")

    # Cross-asset
    print("  Building cross-asset features...")
    ca = cross_asset_features(daily_data)
    vol_feat = volume_features(btc_1d)

    # Seasonality
    print("  Building seasonality features...")
    season = seasonality_features(pd.DatetimeIndex(log_rv.index))

    # Traditional assets (VIX, S&P 500, Gold, DXY, 10Y)
    trad_path = RAW_DIR / "traditional_assets.parquet"
    if trad_path.exists():
        print("  Building traditional asset features...")
        trad_raw = pd.read_parquet(trad_path)
        trad_raw.index = pd.to_datetime(trad_raw.index, utc=True)
        trad_feat = traditional_asset_features(trad_raw)
    else:
        trad_feat = pd.DataFrame()

    # Sentiment
    fg_path = RAW_DIR / "fear_greed.parquet"
    if fg_path.exists():
        fg_raw = pd.read_parquet(fg_path)
        fg_raw["timestamp"] = pd.to_datetime(fg_raw["timestamp"], utc=True)
        fg_feat = sentiment_features(fg_raw)
    else:
        fg_feat = pd.DataFrame()

    # Assemble — align everything to daily UTC index
    frames = [log_rv, har, vov, pct,
              jumps.shift(1),   # shift jumps: use yesterday's jumps to predict today
              semi_var,
              log_gk, season]

    df = pd.concat(frames, axis=1)

    if not ca.empty:
        df = df.join(ca, how="left")
    if not vol_feat.empty:
        df = df.join(vol_feat, how="left")
    if not fg_feat.empty:
        df = df.join(fg_feat, how="left")
    if not trad_feat.empty:
        # Reindex to BTC's daily index then forward-fill weekends/holidays
        trad_feat = trad_feat.reindex(df.index).ffill()
        df = df.join(trad_feat, how="left")

    # Filter date range and drop warmup rows (need 22 days for monthly lag)
    start_ts = pd.Timestamp(start, tz="UTC")
    df = df[df.index >= start_ts]
    df = df.dropna(subset=["rv_1d", "rv_5d", "rv_22d"])  # require HAR lags

    out_path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(out_path)
    print(f"  Saved {len(df):,} rows × {len(df.columns)} cols → {out_path.name}")
    print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Null rate: {df.isnull().mean().mean():.1%}")

    return df


if __name__ == "__main__":
    df = build_feature_matrix()
    print(df.head())
    print(df.dtypes)
