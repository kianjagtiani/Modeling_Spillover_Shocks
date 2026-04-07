"""
Binance.US public API data ingestion.
Fetches OHLCV klines for BTC and cross-asset symbols.
No API key required — uses public endpoints only.

Optimizations vs. v1:
  - Incremental saves every SAVE_EVERY batches (resumable mid-download)
  - Parallel symbol fetching via ThreadPoolExecutor
  - Progress printed to stdout so it's visible even under conda buffering
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

SYMBOLS = {
    "BTCUSDT": "btc",
    "ETHUSDT": "eth",
    "SOLUSDT": "sol",
    "BNBUSDT": "bnb",
}

BINANCE_LIMIT = 1000   # max candles per request
SAVE_EVERY = 50        # save parquet every N batches (resumable)


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
    """
    Fetch klines for a symbol/interval, saving incrementally every SAVE_EVERY batches.
    Resumes from last saved row if parquet already exists.
    """
    end = end or pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    out_path = RAW_DIR / f"{symbol}_{interval}.parquet"

    # Resume from last saved point
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
    batch_num = 0
    total_rows = len(existing)

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": BINANCE_LIMIT,
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

        # Incremental save
        if batch_num % SAVE_EVERY == 0:
            new_df = _parse_klines(accumulated)
            combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
            _save(combined, out_path)
            existing = pd.read_parquet(out_path)
            accumulated = []
            _log(f"  [{symbol} {interval}] checkpoint: {total_rows:,} rows saved")

        if len(data) < BINANCE_LIMIT:
            break

        time.sleep(0.08)

    # Final save
    if accumulated:
        new_df = _parse_klines(accumulated)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        _save(combined, out_path)

    final = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    _log(f"  [{symbol} {interval}] DONE — {len(final):,} rows → {out_path.name}")
    return out_path


TRAD_SYMBOLS = {
    "^VIX":      "vix",        # CBOE Volatility Index — strongest validated predictor
    "^GSPC":     "sp500",      # S&P 500
    "GLD":       "gold",       # Gold ETF
    "DX-Y.NYB":  "dxy",        # US Dollar Index
    "^TNX":      "tnx",        # 10-Year Treasury yield
}


def fetch_traditional_assets(start: str = "2021-01-01") -> pd.DataFrame:
    """
    Fetch daily OHLCV for traditional financial assets via yfinance.
    Returns a single DataFrame indexed by UTC date with all assets.
    """
    import yfinance as yf

    frames = {}
    for sym, name in TRAD_SYMBOLS.items():
        try:
            df = yf.download(sym, start=start, auto_adjust=True, progress=False)
            if df.empty:
                _log(f"  [trad] WARNING: no data for {sym}")
                continue
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].rename(name)
            high  = df["High"].rename(f"{name}_high")
            low   = df["Low"].rename(f"{name}_low")
            frames[name]            = close
            frames[f"{name}_high"]  = high
            frames[f"{name}_low"]   = low
            _log(f"  [trad] {name}: {len(df)} rows")
        except Exception as e:
            _log(f"  [trad] ERROR {sym}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    return out


def fetch_fear_greed(start: str = "2021-01-01") -> pd.DataFrame:
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.DataFrame(data)[["timestamp", "value", "value_classification"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fear_greed"] = df["value"].astype(int)
    df = df[["timestamp", "fear_greed", "value_classification"]]
    df = df.sort_values("timestamp").reset_index(drop=True)
    start_ts = pd.Timestamp(start, tz="UTC")
    return df[df["timestamp"] >= start_ts].reset_index(drop=True)


def ingest_all(start: str = "2021-01-01") -> None:
    """
    Fetch all symbols in parallel (1d and 5m), then Fear & Greed.
    Safe to re-run — resumes from last checkpoint.
    """
    _log("=== Data Ingestion (parallel) ===")

    tasks = [(sym, iv) for sym in SYMBOLS for iv in ["1d", "5m"]]

    # Fetch 1d first (fast), then 5m in parallel
    _log("  Fetching daily data...")
    for sym in SYMBOLS:
        try:
            fetch_and_save(sym, "1d", start=start)
        except Exception as e:
            _log(f"  ERROR {sym} 1d: {e}")

    _log("  Fetching 5-min data (parallel, 4 threads)...")
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

    _log("  Fetching traditional assets (VIX, S&P 500, Gold, DXY, 10Y)...")
    try:
        trad = fetch_traditional_assets(start=start)
        trad.to_parquet(RAW_DIR / "traditional_assets.parquet")
        _log(f"  Saved {len(trad):,} rows → traditional_assets.parquet")
    except Exception as e:
        _log(f"  ERROR traditional assets: {e}")

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
