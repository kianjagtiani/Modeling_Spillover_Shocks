"""
Binance.US public API data ingestion + yfinance for traditional assets.
Expanded to cover 25+ assets across crypto, equities, commodities, and forex.

Crypto (Binance.US, 1d + 5m): BTC, ETH, SOL, BNB
Crypto (Binance.US, 1d only): ADA, DOT, AVAX, UNI, AAVE, MKR
Traditional (yfinance, 1d): VIX, S&P 500, Gold, DXY, 10Y, NDX, DJI, MSCI World,
                             XLK, XLF, SOXX, WTI, Brent, Copper
"""

import time
import sys
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BINANCE_BASE = "https://api.binance.us/api/v3"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Core crypto: fetched at both 1d and 5m resolution
SYMBOLS = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "SOLUSDT": "sol",
    "BNBUSDT": "bnb",
}

# Extended crypto: 1d only (insufficient 5m history / not needed for feature set)
CRYPTO_SYMBOLS_1D_ONLY = {
    "ADAUSDT":  "ada",
    "DOTUSDT":  "dot",
    "AVAXUSDT": "avax",
    "UNIUSDT":  "uni",
    "AAVEUSDT": "aave",
    "MKRUSDT":  "mkr",
}

BINANCE_LIMIT = 1000
SAVE_EVERY    = 50


def _log(msg: str) -> None:
    print(msg, flush=True)


def _parse_klines(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "n_trades"]]
    df = df.rename(columns={"open_time": "timestamp"})
    return df


def _save(df: pd.DataFrame, path: Path) -> None:
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(path, index=False)


def fetch_and_save(
    symbol: str,
    interval: str,
    start: str = "2021-01-01",
    end: str | None = None,
) -> Path:
    """Fetch klines for a symbol/interval, saving incrementally. Resumable."""
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    out_path = RAW_DIR / f"{symbol}_{interval}.parquet"

    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        last_ts = existing["timestamp"].max()
        resume_ms = int(last_ts.timestamp() * 1000) + 1
        if resume_ms >= end_ms:
            _log(f"  [{symbol} {interval}] up to date ({len(existing):,} rows)")
            return out_path
        _log(f"  [{symbol} {interval}] resuming from {last_ts.date()}")
        current_start = resume_ms
    else:
        current_start = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        _log(f"  [{symbol} {interval}] fetching from {start}")

    accumulated = []
    batch_num   = 0
    total_rows  = len(existing)

    while current_start < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current_start,
            "endTime":   end_ms,
            "limit":     BINANCE_LIMIT,
        }
        try:
            resp = requests.get(f"{BINANCE_BASE}/klines", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            _log(f"  [{symbol} {interval}] ERROR batch {batch_num}: {e} — retrying in 5s")
            time.sleep(5)
            continue

        if not data:
            break

        accumulated.extend(data)
        current_start = data[-1][0] + 1
        batch_num += 1
        total_rows += len(data)

        if batch_num % SAVE_EVERY == 0:
            new_df   = _parse_klines(accumulated)
            combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
            _save(combined, out_path)
            existing    = pd.read_parquet(out_path)
            accumulated = []
            _log(f"  [{symbol} {interval}] checkpoint: {total_rows:,} rows saved")

        if len(data) < BINANCE_LIMIT:
            break

        time.sleep(0.08)

    if accumulated:
        new_df   = _parse_klines(accumulated)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        _save(combined, out_path)

    final = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    _log(f"  [{symbol} {interval}] DONE — {len(final):,} rows → {out_path.name}")
    return out_path


# ── Traditional & Cross-Asset (yfinance) ─────────────────────────────────────

TRAD_SYMBOLS = {
    # Original set
    "^VIX":      "vix",        # CBOE Volatility Index
    "^GSPC":     "sp500",      # S&P 500
    "GLD":       "gold",       # Gold ETF
    "DX-Y.NYB":  "dxy",        # US Dollar Index
    "^TNX":      "tnx",        # 10-Year Treasury yield
    # Equity indices
    "^NDX":      "ndx",        # Nasdaq-100
    "^DJI":      "dji",        # Dow Jones Industrial Average
    "URTH":      "msci_world", # MSCI World ETF (iShares)
    # Sector ETFs
    "XLK":       "xlk",        # Technology Select Sector
    "XLF":       "xlf",        # Financial Select Sector
    "SOXX":      "soxx",       # iShares Semiconductor ETF
    # Commodities
    "CL=F":      "wti",        # WTI Crude Oil front-month futures
    "BZ=F":      "brent",      # Brent Crude front-month futures
    "HG=F":      "copper",     # Copper front-month futures
}


def fetch_traditional_assets(start: str = "2021-01-01") -> pd.DataFrame:
    """
    Fetch daily OHLCV for all traditional/macro assets via yfinance.
    Returns a single DataFrame indexed by UTC date.
    Per-symbol errors are caught so one bad ticker never blocks the rest.
    """
    import yfinance as yf

    frames = {}
    for sym, name in TRAD_SYMBOLS.items():
        try:
            df = yf.download(sym, start=start, auto_adjust=True, progress=False)
            if df.empty:
                _log(f"  [trad] WARNING: no data for {sym} ({name})")
                continue

            # Flatten MultiIndex columns produced by some yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close = df["Close"].rename(name)
            frames[name] = close

            if "High" in df.columns and "Low" in df.columns:
                frames[f"{name}_high"] = df["High"].rename(f"{name}_high")
                frames[f"{name}_low"]  = df["Low"].rename(f"{name}_low")

            _log(f"  [trad] {name:>14} ({sym}): {len(df)} rows")
        except Exception as e:
            _log(f"  [trad] ERROR {sym} ({name}): {e}")

    if not frames:
        _log("  [trad] WARNING: no traditional asset data fetched")
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    return out


def fetch_crypto_1d_only(
    symbols: dict | None = None,
    start: str = "2021-01-01",
) -> None:
    """
    Fetch daily (1d) klines for extended crypto assets that don't need 5m data.
    symbols: dict of {SYMBOL: name} — defaults to CRYPTO_SYMBOLS_1D_ONLY.
    """
    syms = symbols or CRYPTO_SYMBOLS_1D_ONLY
    _log(f"  Fetching 1d-only crypto ({len(syms)} symbols)...")
    for sym in syms:
        try:
            fetch_and_save(sym, "1d", start=start)
        except Exception as e:
            _log(f"  ERROR {sym} 1d: {e}")


def fetch_fear_greed(start: str = "2021-01-01") -> pd.DataFrame:
    url  = "https://api.alternative.me/fng/?limit=0&format=json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]
    df   = pd.DataFrame(data)[["timestamp", "value", "value_classification"]]
    df["timestamp"]  = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fear_greed"] = df["value"].astype(int)
    df = df[["timestamp", "fear_greed", "value_classification"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    return df[df["timestamp"] >= start_ts].reset_index(drop=True)


def ingest_all(start: str = "2021-01-01") -> None:
    """
    Fetch all symbols (parallel where possible), then Fear & Greed.
    Safe to re-run — resumes from last checkpoint.
    """
    _log("=== Data Ingestion (expanded) ===")

    # ── Core crypto: 1d first (fast), then 5m in parallel ────────────────────
    _log("  Fetching core crypto daily data...")
    for sym in SYMBOLS:
        try:
            fetch_and_save(sym, "1d", start=start)
        except Exception as e:
            _log(f"  ERROR {sym} 1d: {e}")

    _log("  Fetching core crypto 5-min data (parallel, 4 threads)...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_and_save, sym, "5m", start): sym
            for sym in SYMBOLS
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                fut.result()
            except Exception as e:
                _log(f"  ERROR {sym} 5m: {e}")

    # ── Extended crypto: 1d only ──────────────────────────────────────────────
    fetch_crypto_1d_only(start=start)

    # ── Traditional assets (yfinance) ─────────────────────────────────────────
    _log(f"  Fetching {len(TRAD_SYMBOLS)} traditional/macro assets via yfinance...")
    try:
        trad = fetch_traditional_assets(start=start)
        if not trad.empty:
            trad.to_parquet(RAW_DIR / "traditional_assets.parquet")
            _log(f"  Saved {len(trad):,} rows × {len(trad.columns)} cols → traditional_assets.parquet")
        else:
            _log("  WARNING: traditional assets DataFrame is empty; skipping save")
    except Exception as e:
        _log(f"  ERROR traditional assets: {e}")

    # ── Fear & Greed ──────────────────────────────────────────────────────────
    _log("  Fetching Fear & Greed Index...")
    try:
        fg = fetch_fear_greed(start=start)
        fg.to_parquet(RAW_DIR / "fear_greed.parquet", index=False)
        _log(f"  Saved {len(fg):,} rows → fear_greed.parquet")
    except Exception as e:
        _log(f"  ERROR Fear & Greed: {e}")

    _log("=== Ingestion complete ===")


if __name__ == "__main__":
    start_date = sys.argv[1] if len(sys.argv) > 1 else "2021-01-01"
    ingest_all(start=start_date)
