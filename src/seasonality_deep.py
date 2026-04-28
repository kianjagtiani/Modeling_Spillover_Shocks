"""
Enhanced seasonality and macro calendar features for BTC volatility forecasting.

Extends the basic seasonality in features.py with:
  - CME BTC/ETH futures and options expiry dates
  - FOMC meeting dates (key macro catalyst)
  - CPI release windows
  - Bitcoin halving cycle granularity
  - Intraweek patterns (Monday gap, Friday settlement, Sunday low-liquidity)
  - Quarter-end and month-end rebalancing signals
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── FOMC Meeting Dates 2021-2026 ──────────────────────────────────────────────
# Source: Federal Reserve published calendars
FOMC_DATES = pd.to_datetime([
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-02-02", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18",
], utc=True)

# ── Bitcoin Halving Dates ─────────────────────────────────────────────────────
HALVING_DATES = pd.to_datetime([
    "2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20"
], utc=True)
# Projected next halving (~April 2028)
NEXT_HALVING = pd.Timestamp("2028-04-14", tz="UTC")


# ── Helper: last weekday of a given weekday in a month ────────────────────────

def _last_weekday_of_month(year: int, month: int, weekday: int = 4) -> pd.Timestamp:
    """
    Return the last occurrence of weekday (0=Mon … 4=Fri) in the given month.
    weekday=4 → last Friday.
    """
    # Start from last day of month and walk backward
    last_day = pd.Timestamp(year=year, month=month,
                            day=pd.Timestamp(year, month, 1).days_in_month, tz="UTC")
    delta = (last_day.weekday() - weekday) % 7
    return last_day - pd.Timedelta(days=int(delta))


def _build_cme_expiry_dates(
    start_year: int = 2021, end_year: int = 2027
) -> tuple[list, list]:
    """
    Return (quarterly_expiries, monthly_expiries) as sorted lists of Timestamps.
    CME BTC/ETH derivatives expire on the last Friday of the expiry month.
    Quarterly: March, June, September, December.
    Monthly: every month.
    """
    quarterly, monthly = [], []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            last_fri = _last_weekday_of_month(year, month, weekday=4)
            monthly.append(last_fri)
            if month in (3, 6, 9, 12):
                quarterly.append(last_fri)
    return sorted(quarterly), sorted(monthly)


_QUARTERLY_EXPIRIES, _MONTHLY_EXPIRIES = _build_cme_expiry_dates()
_QUARTERLY_SET = set(ts.date() for ts in _QUARTERLY_EXPIRIES)
_MONTHLY_SET   = set(ts.date() for ts in _MONTHLY_EXPIRIES)


# ── 1. CME Expiry Features ────────────────────────────────────────────────────

def cme_expiry_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Distance-to-expiry features for CME BTC/ETH futures and options.
    Vol historically spikes in the 5 days before quarterly expiry.
    """
    dates   = pd.Series(pd.to_datetime(index).normalize(), index=index)
    records = []

    quarterly_ts = pd.Series(_QUARTERLY_EXPIRIES)
    monthly_ts   = pd.Series(_MONTHLY_EXPIRIES)

    for date in dates:
        d = date.date()

        # Days to next quarterly expiry
        future_q = quarterly_ts[quarterly_ts >= date]
        days_to_q = int((future_q.iloc[0] - date).days) if len(future_q) > 0 else 92

        # Days since last quarterly expiry
        past_q = quarterly_ts[quarterly_ts <= date]
        days_since_q = int((date - past_q.iloc[-1]).days) if len(past_q) > 0 else 92

        # Days to next monthly expiry
        future_m = monthly_ts[monthly_ts >= date]
        days_to_m = int((future_m.iloc[0] - date).days) if len(future_m) > 0 else 32

        # Proximity score: peaks sharply at expiry, decays over 7 days
        cme_q_score = float(np.exp(-days_to_q / 3.0)) if days_to_q <= 10 else 0.0
        cme_m_score = float(np.exp(-days_to_m / 2.0)) if days_to_m <= 5 else 0.0

        records.append({
            "days_to_cme_quarterly":  min(days_to_q, 92),
            "days_since_cme_quarterly": min(days_since_q, 92),
            "days_to_cme_monthly":    min(days_to_m, 32),
            "is_cme_quarterly_week":  int(days_to_q <= 5 or days_since_q <= 2),
            "is_cme_monthly_week":    int(days_to_m <= 3),
            "is_cme_expiry_friday":   int(d in _QUARTERLY_SET or d in _MONTHLY_SET),
            "cme_q_proximity_score":  round(cme_q_score, 4),
            "cme_m_proximity_score":  round(cme_m_score, 4),
        })

    df = pd.DataFrame(records, index=index)
    # Cyclic encoding of days-to-quarterly for smooth model input
    df["cme_q_sin"] = np.sin(2 * np.pi * df["days_to_cme_quarterly"] / 92)
    df["cme_q_cos"] = np.cos(2 * np.pi * df["days_to_cme_quarterly"] / 92)
    return df


# ── 2. Macro Calendar Features ────────────────────────────────────────────────

def macro_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    FOMC meeting proximity and CPI release window features.
    FOMC days consistently elevate BTC volatility as liquidity providers
    widen spreads ahead of the announcement.
    """
    dates    = pd.to_datetime(index).normalize()
    fomc_set = set(ts.normalize().date() for ts in FOMC_DATES)

    records = []
    for ts in dates:
        d = ts.date()

        # Days to next FOMC
        future_fomc = FOMC_DATES[FOMC_DATES >= ts]
        days_to_fomc = int((future_fomc[0] - ts).days) if len(future_fomc) > 0 else 45

        # Days since last FOMC
        past_fomc = FOMC_DATES[FOMC_DATES <= ts]
        days_since_fomc = int((ts - past_fomc[-1]).days) if len(past_fomc) > 0 else 45

        # Proximity score: high in 3-day window around FOMC
        fomc_proximity = float(np.exp(-min(days_to_fomc, days_since_fomc) / 2.0))

        # CPI release: approximately days 8-14 of each month
        is_cpi_week = int(8 <= ts.day <= 15)

        records.append({
            "days_to_fomc":       min(days_to_fomc, 45),
            "days_since_fomc":    min(days_since_fomc, 45),
            "is_fomc_week":       int(days_to_fomc <= 3 or days_since_fomc <= 1),
            "is_fomc_day":        int(d in fomc_set),
            "fomc_proximity_score": round(fomc_proximity, 4),
            "is_cpi_week":        is_cpi_week,
        })

    df = pd.DataFrame(records, index=index)
    df["fomc_sin"] = np.sin(2 * np.pi * df["days_to_fomc"] / 45)
    df["fomc_cos"] = np.cos(2 * np.pi * df["days_to_fomc"] / 45)
    return df


# ── 3. Bitcoin Halving Cycle Features ─────────────────────────────────────────

def btc_cycle_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Features encoding position within Bitcoin's 4-year halving cycle.
    Vol historically suppressed in Year 1 post-halving, elevated before halvings.
    """
    dates   = pd.to_datetime(index).normalize()
    records = []
    all_halvings = list(HALVING_DATES) + [NEXT_HALVING]

    for ts in dates:
        # Last past halving
        past = [h for h in all_halvings if h <= ts]
        last_halving = past[-1] if past else all_halvings[0]

        # Next halving
        future = [h for h in all_halvings if h > ts]
        next_halving = future[0] if future else NEXT_HALVING

        days_since = int((ts - last_halving).days)
        days_to_next = int((next_halving - ts).days)

        # Cycle day (0-1460) within current 4-year cycle
        cycle_day = days_since % 1461

        # Post-halving year flags
        post_halving_year1 = int(days_since <= 365)
        post_halving_year2 = int(366 <= days_since <= 730)
        post_halving_year3 = int(731 <= days_since <= 1095)
        pre_halving_90d    = int(days_to_next <= 90)
        pre_halving_30d    = int(days_to_next <= 30)

        # Quarter within 4-year cycle (0-15)
        cycle_quarter = cycle_day // 91

        records.append({
            "halving_cycle_day":         cycle_day,
            "days_since_halving":        min(days_since, 1461),
            "days_to_next_halving":      min(days_to_next, 1461),
            "post_halving_year1":        post_halving_year1,
            "post_halving_year2":        post_halving_year2,
            "post_halving_year3":        post_halving_year3,
            "pre_halving_90d":           pre_halving_90d,
            "pre_halving_30d":           pre_halving_30d,
            "halving_cycle_quarter":     cycle_quarter,
        })

    df = pd.DataFrame(records, index=index)
    df["halving_cycle_sin"] = np.sin(2 * np.pi * df["halving_cycle_day"] / 1461)
    df["halving_cycle_cos"] = np.cos(2 * np.pi * df["halving_cycle_day"] / 1461)
    return df


# ── 4. Intraweek Features ─────────────────────────────────────────────────────

def intraweek_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Day-of-week patterns validated in crypto literature.
    Sunday: lowest liquidity, highest vol per unit move.
    Monday: gap risk from weekend.
    Friday: CME settlement, position squaring.
    """
    dates = pd.to_datetime(index)
    dow   = dates.dayofweek  # 0=Mon … 6=Sun
    week_of_month = ((dates.day - 1) // 7 + 1).astype(int)

    return pd.DataFrame({
        "is_monday":      (dow == 0).astype(int),
        "is_tuesday":     (dow == 1).astype(int),
        "is_wednesday":   (dow == 2).astype(int),
        "is_thursday":    (dow == 3).astype(int),
        "is_friday":      (dow == 4).astype(int),
        "is_saturday":    (dow == 5).astype(int),
        "is_sunday":      (dow == 6).astype(int),
        "week_of_month":  week_of_month,
    }, index=index)


# ── 5. Quarterly Rebalancing Features ────────────────────────────────────────

def quarterly_rebalancing_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    End-of-month and end-of-quarter rebalancing creates vol as institutions
    unwind or re-risk positions to meet mandate allocations.
    """
    dates = pd.to_datetime(index)

    # Days to end of current month
    def days_to_month_end(ts):
        month_end = (ts + pd.offsets.MonthEnd(0))
        return int((month_end - ts).days)

    dtme = pd.Series([days_to_month_end(ts) for ts in dates], index=index)

    is_month_end        = (dtme <= 3).astype(int)
    is_quarter_end_week = (dates.is_quarter_end | dtme.le(5)).astype(int)
    is_year_end         = ((dates.month == 12) & (dates.day >= 28) |
                           (dates.month == 1) & (dates.day <= 3)).astype(int)
    is_month_start      = (dates.day <= 3).astype(int)

    return pd.DataFrame({
        "days_to_month_end":     dtme.clip(upper=31),
        "is_month_end":          is_month_end,
        "is_month_start":        is_month_start,
        "is_quarter_end_week":   is_quarter_end_week,
        "is_year_end":           is_year_end,
    }, index=index)


# ── 6. Combined Builder ───────────────────────────────────────────────────────

def build_deep_seasonality(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Combine all seasonality feature groups into one DataFrame.
    """
    frames = [
        cme_expiry_features(index),
        macro_calendar_features(index),
        btc_cycle_features(index),
        intraweek_features(index),
        quarterly_rebalancing_features(index),
    ]
    return pd.concat(frames, axis=1)


# ── 7. Integration Helper ─────────────────────────────────────────────────────

def add_seasonality_to_features(features_path: Path | None = None) -> pd.DataFrame:
    """
    Load features.parquet, join deep seasonality features, and save to
    data/processed/features_seasonality.parquet.
    """
    path = features_path or (PROCESSED_DIR / "features.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {path}")

    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)

    n_before = len(df.columns)
    season   = build_deep_seasonality(pd.DatetimeIndex(df.index))
    df       = df.join(season, how="left")
    n_after  = len(df.columns)

    out_path = PROCESSED_DIR / "features_seasonality.parquet"
    df.to_parquet(out_path)
    print(f"Deep seasonality: {n_before} → {n_after} features (+{n_after - n_before})")
    print(f"Saved → {out_path}")
    return df


if __name__ == "__main__":
    add_seasonality_to_features()
