"""
Options strategy module for BTC volatility-based trading.

Core insight: if our model accurately forecasts realized volatility,
we can profit when the options market misprices it (volatility risk premium).

Deribit is the dominant BTC options exchange (~80% market share). Typical IV
trades at a 10-20% premium over subsequent realized vol (the VRP). Our model,
which forecasts realized vol, can identify when this premium is unusually
large (sell vol) or unusually small (buy vol).

Simulation note: since we can't access live Deribit data, we simulate
implied vol as:  IV = forecast_rv * (1 + vol_premium) + noise
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import norm

warnings.filterwarnings("ignore")

RAW_DIR     = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Black-Scholes Pricer (Crypto-Adapted) ─────────────────────────────────

class BlackScholes:
    """
    Black-Scholes option pricer adapted for 24/7 crypto markets.
    T is expressed in calendar days / 365 (not trading days / 252).
    r = 0 (BTC has no clean risk-free rate; funding rates are small).
    """

    def __init__(self, S: float, K: float, T_days: float, sigma: float,
                 r: float = 0.0, option_type: str = "call"):
        """
        S:           Spot price
        K:           Strike price
        T_days:      Days to expiry (calendar days, crypto trades 24/7)
        sigma:       Annual volatility (e.g. 0.50 = 50% annualized)
        r:           Risk-free rate (default 0 for BTC)
        option_type: 'call' or 'put'
        """
        self.S    = S
        self.K    = K
        self.T    = max(T_days / 365.0, 1e-6)
        self.sigma = max(sigma, 1e-6)
        self.r    = r
        self.option_type = option_type.lower()

    def _d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / \
             (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self) -> float:
        d1, d2 = self._d1_d2()
        if self.option_type == "call":
            return float(self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:
            return float(self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1))

    def delta(self) -> float:
        d1, _ = self._d1_d2()
        if self.option_type == "call":
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        return float(norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T)))

    def vega(self) -> float:
        """Vega: change in option price per 1 percentage point move in vol."""
        d1, _ = self._d1_d2()
        return float(self.S * norm.pdf(d1) * np.sqrt(self.T) * 0.01)

    def theta(self) -> float:
        """Theta: daily time decay (option loses theta per calendar day)."""
        d1, d2 = self._d1_d2()
        term1 = -(self.S * norm.pdf(d1) * self.sigma) / (2 * np.sqrt(self.T))
        if self.option_type == "call":
            term2 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2)
        return float((term1 + term2) / 365)

    @classmethod
    def implied_vol(
        cls,
        market_price: float,
        S: float,
        K: float,
        T_days: float,
        r: float = 0.0,
        option_type: str = "call",
        tol: float = 1e-6,
        max_iter: int = 100,
    ) -> float:
        """Newton-Raphson IV solver with bisection fallback."""
        sigma = 0.5  # initial guess
        for _ in range(max_iter):
            bs   = cls(S, K, T_days, sigma, r, option_type)
            diff = bs.price() - market_price
            if abs(diff) < tol:
                return sigma
            vega = bs.vega() * 100  # convert back from per-1% to per-unit
            if abs(vega) < 1e-10:
                break
            sigma -= diff / vega
            sigma = max(0.001, min(sigma, 20.0))

        # Bisection fallback
        lo, hi = 0.001, 20.0
        for _ in range(200):
            mid = (lo + hi) / 2
            val = cls(S, K, T_days, mid, r, option_type).price() - market_price
            if abs(val) < tol:
                return mid
            if val > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2


# ── 2. Options Position ───────────────────────────────────────────────────────

@dataclass
class OptionsPosition:
    """Track a simulated options position."""
    entry_date:       pd.Timestamp
    expiry_date:      pd.Timestamp
    strike:           float
    option_type:      str           # 'call', 'put', or 'straddle'
    entry_vol:        float         # IV when we entered
    entry_price:      float         # option price paid/received
    quantity:         float         # +1 = long, -1 = short
    underlying_entry: float         # BTC price at entry
    position_size_usd: float = 1000.0

    def days_to_expiry(self, current_date: pd.Timestamp) -> float:
        return max((self.expiry_date - current_date).days, 0)

    def current_value(self, S: float, sigma: float, current_date: pd.Timestamp) -> float:
        T = self.days_to_expiry(current_date)
        if self.option_type == "straddle":
            call = BlackScholes(S, self.strike, T, sigma, option_type="call").price()
            put  = BlackScholes(S, self.strike, T, sigma, option_type="put").price()
            return (call + put) * self.quantity
        else:
            return BlackScholes(S, self.strike, T, sigma, option_type=self.option_type).price() * self.quantity

    def pnl(self, S: float, sigma: float, current_date: pd.Timestamp) -> float:
        return (self.current_value(S, sigma, current_date) - self.entry_price * self.quantity)

    def intrinsic_pnl_at_expiry(self, S_expiry: float) -> float:
        """P&L at expiration based on intrinsic value only."""
        if self.option_type == "call":
            intrinsic = max(S_expiry - self.strike, 0)
        elif self.option_type == "put":
            intrinsic = max(self.strike - S_expiry, 0)
        elif self.option_type == "straddle":
            intrinsic = max(S_expiry - self.strike, 0) + max(self.strike - S_expiry, 0)
        else:
            intrinsic = 0.0
        return (intrinsic * self.quantity) - (self.entry_price * self.quantity)


# ── 3. Vol Forecast Options Strategy ─────────────────────────────────────────

class VolForecastOptionsStrategy:
    """
    Trade the volatility risk premium using our realized vol forecasts.

    Logic:
      - If model_vol > simulated_IV * (1 + threshold):
            BUY straddle (we expect bigger moves than market prices)
      - If model_vol < simulated_IV * (1 - threshold):
            SELL straddle (we expect smaller moves, collect the premium)
      - Otherwise: no trade

    The simulated IV is: IV = model_forecast * (1 + vol_premium) + noise
    This reflects the typical 10-20% Deribit premium over subsequent realized vol.
    """

    def __init__(
        self,
        vol_premium_threshold: float = 0.15,
        position_size_usd:     float = 1000.0,
        expiry_days:           int   = 7,
        tx_cost_pct:           float = 0.05,
        max_vega_usd:          float = 500.0,
    ):
        self.threshold        = vol_premium_threshold
        self.position_size    = position_size_usd
        self.expiry_days      = expiry_days
        self.tx_cost_pct      = tx_cost_pct
        self.max_vega_usd     = max_vega_usd

    def generate_signal(
        self,
        forecast_vol: float,
        implied_vol:  float,
    ) -> dict:
        """Determine whether to buy/sell straddle or do nothing."""
        vol_edge     = forecast_vol - implied_vol
        vol_edge_pct = vol_edge / max(implied_vol, 1e-6)

        if vol_edge_pct > self.threshold:
            action = "buy_straddle"     # model says vol will be higher than market prices
        elif vol_edge_pct < -self.threshold:
            action = "sell_straddle"    # model says vol will be lower — collect premium
        else:
            action = "neutral"

        return {
            "action":       action,
            "vol_edge":     vol_edge,
            "vol_edge_pct": vol_edge_pct,
            "confidence":   abs(vol_edge_pct) / self.threshold,
        }

    def size_position(self, vol_edge_pct: float, option_vega: float) -> float:
        """
        Scale position size by confidence in the edge.
        Kelly-inspired: larger edge → larger position, capped by vega limit.
        """
        kelly_fraction = min(abs(vol_edge_pct) / self.threshold, 3.0) / 3.0
        desired_usd    = self.position_size * kelly_fraction
        # Vega limit: don't take more than max_vega_usd of vega exposure
        if option_vega > 0:
            max_by_vega = self.max_vega_usd / (option_vega * 100)  # vega per 100 units
            desired_usd = min(desired_usd, max_by_vega * 100)
        return max(desired_usd, 10.0)

    def select_strike(self, S: float, sigma: float, T_days: float,
                      target_delta: float = 0.50) -> float:
        """Select ATM strike for straddles (delta = 0.50 ≈ ATM)."""
        # ATM: strike = spot (for 0-delta straddle)
        return S

    def backtest_options_strategy(
        self,
        btc_prices:      pd.Series,
        predicted_vol:   pd.Series,   # annualized, from our model
        vol_premium:     float = 0.15,
        rng_seed:        int   = 42,
    ) -> pd.DataFrame:
        """
        Walk-forward backtest of the options strategy.

        Simulates Deribit-style weekly ATM straddles (7-day expiry).
        At each step:
          1. Compute simulated IV = our_forecast * (1 + vol_premium) + noise
          2. Generate trade signal
          3. Price the straddle
          4. Enter / exit position
          5. Record P&L at expiry

        Returns DataFrame with daily P&L.
        """
        rng = np.random.default_rng(rng_seed)
        idx = btc_prices.index.intersection(predicted_vol.index)
        prices = btc_prices.loc[idx]
        preds  = predicted_vol.loc[idx]

        records = []
        position: OptionsPosition | None = None

        for i, date in enumerate(idx):
            S        = float(prices.loc[date])
            forecast = float(preds.loc[date])

            # Simulate market IV (what Deribit is quoting)
            noise = rng.normal(0, vol_premium * 0.2)
            sim_iv = forecast * (1 + vol_premium) + noise
            sim_iv = max(sim_iv, 0.10)  # floor at 10% annualized

            # Check open position
            day_pnl = 0.0
            if position is not None:
                if date >= position.expiry_date:
                    # Settle at expiry (intrinsic value)
                    day_pnl  = position.intrinsic_pnl_at_expiry(S)
                    # Scale to position size in USD
                    day_pnl_usd = day_pnl / position.entry_price * position.position_size_usd
                    records.append({
                        "date":        date,
                        "action":      "expiry",
                        "pnl_usd":     day_pnl_usd,
                        "vol_forecast": forecast,
                        "sim_iv":      sim_iv,
                        "vol_edge":    forecast - sim_iv,
                        "btc_price":   S,
                    })
                    position = None

            # Consider new position every 7 days (or on open position expiry)
            if position is None and i % self.expiry_days == 0:
                signal = self.generate_signal(forecast, sim_iv)

                if signal["action"] != "neutral":
                    K = self.select_strike(S, sim_iv, self.expiry_days)
                    call_bs = BlackScholes(S, K, self.expiry_days, sim_iv, option_type="call")
                    put_bs  = BlackScholes(S, K, self.expiry_days, sim_iv, option_type="put")
                    straddle_price = call_bs.price() + put_bs.price()
                    straddle_vega  = call_bs.vega() + put_bs.vega()

                    if straddle_price <= 0:
                        continue

                    qty_sign = 1.0 if signal["action"] == "buy_straddle" else -1.0
                    pos_size = self.size_position(signal["vol_edge_pct"], straddle_vega)

                    # Transaction cost
                    tx_cost = straddle_price * abs(qty_sign) * self.tx_cost_pct

                    expiry_date = date + pd.Timedelta(days=self.expiry_days)
                    # Snap to a date in our index if possible
                    future_dates = idx[idx >= expiry_date]
                    if len(future_dates) == 0:
                        continue
                    actual_expiry = future_dates[0]

                    position = OptionsPosition(
                        entry_date=date,
                        expiry_date=actual_expiry,
                        strike=K,
                        option_type="straddle",
                        entry_vol=sim_iv,
                        entry_price=straddle_price + tx_cost / abs(qty_sign),
                        quantity=qty_sign,
                        underlying_entry=S,
                        position_size_usd=pos_size,
                    )

                    records.append({
                        "date":         date,
                        "action":       signal["action"],
                        "pnl_usd":      -tx_cost,  # cost of entry
                        "vol_forecast": forecast,
                        "sim_iv":       sim_iv,
                        "vol_edge":     signal["vol_edge"],
                        "btc_price":    S,
                    })

        result = pd.DataFrame(records).set_index("date")
        if not result.empty:
            result["cumulative_pnl"] = result["pnl_usd"].cumsum()
        return result

    def run(
        self,
        predictions_path: Path | None = None,
        btc_prices_path:  Path | None = None,
    ) -> dict:
        """Load model predictions, run backtest, save results."""
        pred_path = predictions_path or (Path(__file__).parent.parent /
                                         "results" / "preds_har_lgbm_hybrid_h1.parquet")
        btc_path  = btc_prices_path  or (Path(__file__).parent.parent /
                                          "data" / "raw" / "BTCUSDT_1d.parquet")

        if not pred_path.exists():
            print(f"Predictions file not found: {pred_path}")
            print("Run pipeline.py first to generate model predictions.")
            return {}

        preds = pd.read_parquet(pred_path)
        preds.index = pd.to_datetime(preds.index, utc=True)
        forecast_rv = np.exp(preds["predicted"])   # convert from log-RV to RV

        btc = pd.read_parquet(btc_path)
        btc["timestamp"] = pd.to_datetime(btc["timestamp"], utc=True)
        btc = btc.set_index("timestamp").sort_index()
        btc_prices = btc["close"]

        results_df = self.backtest_options_strategy(btc_prices, forecast_rv)

        if results_df.empty:
            print("No trades generated.")
            return {}

        out_path = RESULTS_DIR / "options_strategy_results.csv"
        results_df.to_csv(out_path)
        print(f"Saved → {out_path.name}")

        summary = options_strategy_summary(results_df)
        print(summary)

        plot_options_results(results_df, RESULTS_DIR / "options_strategy_chart.png")

        return {
            "results":  results_df,
            "summary":  summary,
        }


# ── 4. Visualization ──────────────────────────────────────────────────────────

def plot_options_results(results_df: pd.DataFrame, output_path: Path) -> None:
    """2-panel chart: cumulative P&L + vol edge over time."""
    if results_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("BTC Volatility-Based Options Strategy", fontsize=13, fontweight="bold")

    # Panel 1: cumulative P&L
    if "cumulative_pnl" in results_df.columns:
        ax1.plot(results_df.index, results_df["cumulative_pnl"],
                 color="#2ecc71", linewidth=1.5, label="Cumulative P&L (USD)")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.fill_between(results_df.index, results_df["cumulative_pnl"], 0,
                         where=results_df["cumulative_pnl"] >= 0,
                         alpha=0.3, color="#2ecc71")
        ax1.fill_between(results_df.index, results_df["cumulative_pnl"], 0,
                         where=results_df["cumulative_pnl"] < 0,
                         alpha=0.3, color="#e74c3c")
    ax1.set_ylabel("Cumulative P&L (USD)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)

    # Panel 2: vol edge (model forecast - simulated IV)
    if "vol_edge" in results_df.columns:
        ax2.bar(results_df.index, results_df["vol_edge"] * 100,
                color=np.where(results_df["vol_edge"] > 0, "#e74c3c", "#3498db"),
                alpha=0.6, width=2, label="Vol edge = Forecast - Sim IV (%)")
        ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_ylabel("Vol Edge (annualized %)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {output_path.name}")
    plt.close()


# ── 5. Summary ────────────────────────────────────────────────────────────────

def options_strategy_summary(results_df: pd.DataFrame) -> str:
    """Format a performance summary string."""
    if results_df.empty:
        return "No results to summarize."

    trades = results_df[results_df["action"].isin(["buy_straddle", "sell_straddle",
                                                    "expiry"])]
    expiry = results_df[results_df["action"] == "expiry"]

    total_pnl    = results_df["pnl_usd"].sum()
    n_trades     = len(results_df[results_df["action"].isin(["buy_straddle", "sell_straddle"])])
    n_winners    = len(expiry[expiry["pnl_usd"] > 0])
    n_expiries   = len(expiry)
    win_rate     = n_winners / n_expiries if n_expiries > 0 else 0
    avg_win      = expiry[expiry["pnl_usd"] > 0]["pnl_usd"].mean() if n_winners > 0 else 0
    avg_loss     = expiry[expiry["pnl_usd"] <= 0]["pnl_usd"].mean() if (n_expiries - n_winners) > 0 else 0
    best_trade   = expiry["pnl_usd"].max() if n_expiries > 0 else 0
    worst_trade  = expiry["pnl_usd"].min() if n_expiries > 0 else 0

    daily_pnl    = results_df.groupby("date")["pnl_usd"].sum()
    sharpe       = daily_pnl.mean() / daily_pnl.std() * np.sqrt(365) if daily_pnl.std() > 0 else 0

    return "\n".join([
        "=" * 55,
        "OPTIONS STRATEGY PERFORMANCE SUMMARY",
        "=" * 55,
        f"  Total P&L:        ${total_pnl:>10,.2f}",
        f"  Sharpe Ratio:     {sharpe:>10.3f} (annualized)",
        f"  Total Trades:     {n_trades:>10}",
        f"  Settled Trades:   {n_expiries:>10}",
        f"  Win Rate:         {win_rate:>10.1%}",
        f"  Avg Win:          ${avg_win:>10,.2f}",
        f"  Avg Loss:         ${avg_loss:>10,.2f}",
        f"  Best Trade:       ${best_trade:>10,.2f}",
        f"  Worst Trade:      ${worst_trade:>10,.2f}",
        "=" * 55,
    ])


if __name__ == "__main__":
    strategy = VolForecastOptionsStrategy(
        vol_premium_threshold=0.15,
        position_size_usd=1000,
        expiry_days=7,
    )
    strategy.run()
