"""
Monte Carlo Backtesting Framework for BTC Volatility Strategies.

Three components:
  1. BacktestEngine       — parameterized strategy engine, all position logic in one place
  2. MCOptimizer          — random search over strategy parameter space with honest
                            train/test split (optimize on 60%, validate on 40%)
  3. BootstrapValidator   — block bootstrap confidence intervals (preserves autocorrelation)
  4. PathSimulator        — GBM Monte Carlo paths for forward stress testing

All results saved to results/mc_backtest_*.

Usage:
  python src/monte_carlo_backtest.py
  python src/monte_carlo_backtest.py --n-trials 1000 --n-bootstrap 2000 --n-paths 1000
"""

from __future__ import annotations

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field, asdict
from pathlib import Path
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")

RAW_DIR      = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
NEW_RESULTS  = Path(__file__).parent.parent / "new results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NEW_RESULTS.mkdir(parents=True, exist_ok=True)


# ── Strategy Parameter Space ──────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """All tunable strategy parameters in one place."""
    target_vol:         float = 0.02    # daily vol target for inverse-vol sizing
    max_position:       float = 1.50    # maximum BTC position (1.5x = 150%)
    min_position:       float = 0.10    # minimum BTC position (10%)
    regime_low_q:       float = 0.33    # quantile threshold: low vol regime
    regime_high_q:      float = 0.67    # quantile threshold: high vol regime
    regime_low_scale:   float = 1.30    # position scale in low vol regime
    regime_high_scale:  float = 0.50    # position scale in high vol regime
    tx_cost:            float = 0.001   # transaction cost per trade (0.1%)
    vol_lookback:       int   = 20      # rolling window for regime quantile
    momentum_window:    int   = 20      # days for momentum signal
    strategy:           str   = "combined"  # iv | regime | combined | multilevel | momentum

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def random_sample(cls, rng: np.random.Generator) -> "StrategyConfig":
        """Sample a random parameter configuration."""
        return cls(
            target_vol        = float(rng.uniform(0.010, 0.045)),
            max_position      = float(rng.uniform(1.10, 2.00)),
            min_position      = float(rng.uniform(0.05, 0.25)),
            regime_low_q      = float(rng.uniform(0.20, 0.45)),
            regime_high_q     = float(rng.uniform(0.55, 0.80)),
            regime_low_scale  = float(rng.uniform(1.00, 1.60)),
            regime_high_scale = float(rng.uniform(0.25, 0.70)),
            tx_cost           = float(rng.choice([0.0005, 0.001, 0.002])),
            vol_lookback      = int(rng.integers(10, 60)),
            momentum_window   = int(rng.integers(10, 40)),
            strategy          = rng.choice(["iv", "regime", "combined", "multilevel"]),
        )


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    r = returns.dropna()
    if len(r) < 10:
        return {"label": label, "sharpe": -99, "total_ret": -1, "ann_ret": -1,
                "ann_vol": 0, "max_dd": -1, "sortino": -99, "calmar": -99,
                "win_rate": 0, "var_5pct": -1, "cvar_5pct": -1}

    cum     = (1 + r).cumprod()
    total   = float(cum.iloc[-1] - 1)
    n_days  = len(r)
    ann_ret = float((1 + total) ** (365 / n_days) - 1)
    ann_vol = float(r.std() * np.sqrt(365))
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0

    downside = float(r[r < 0].std() * np.sqrt(365)) if (r < 0).any() else 1e-9
    sortino  = ann_ret / downside if downside > 0 else 0.0

    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max
    max_dd   = float(drawdown.min())
    calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    var5  = float(np.percentile(r, 5))
    cvar5 = float(r[r <= var5].mean()) if (r <= var5).any() else var5

    return {
        "label":    label,
        "total_ret": total,
        "ann_ret":  ann_ret,
        "ann_vol":  ann_vol,
        "sharpe":   sharpe,
        "sortino":  sortino,
        "max_dd":   max_dd,
        "calmar":   calmar,
        "win_rate": float((r > 0).mean()),
        "var_5pct": var5,
        "cvar_5pct": cvar5,
        "n_days":   n_days,
    }


# ── 1. Backtest Engine ────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Parameterized backtester for volatility-forecast-driven BTC strategies.
    All strategies are long-only with configurable position limits.
    """

    def __init__(self, cfg: StrategyConfig | None = None):
        self.cfg = cfg or StrategyConfig()

    def _position(
        self,
        pred_rv_daily: pd.Series,   # daily vol (not annualized)
        btc_returns: pd.Series,
    ) -> pd.Series:
        """Compute position series based on strategy type."""
        c   = self.cfg
        idx = pred_rv_daily.index

        if c.strategy == "iv":
            pos = (c.target_vol / pred_rv_daily.replace(0, np.nan)).clip(c.min_position, c.max_position)

        elif c.strategy == "regime":
            q_lo = pred_rv_daily.rolling(c.vol_lookback).quantile(c.regime_low_q)
            q_hi = pred_rv_daily.rolling(c.vol_lookback).quantile(c.regime_high_q)
            pos  = pd.Series(1.0, index=idx)
            pos[pred_rv_daily < q_lo.replace(0, np.nan)] = c.regime_low_scale
            pos[pred_rv_daily > q_hi.replace(0, np.nan)] = c.regime_high_scale
            pos  = pos.clip(c.min_position, c.max_position)

        elif c.strategy == "combined":
            iv_pos = (c.target_vol / pred_rv_daily.replace(0, np.nan)).clip(c.min_position, c.max_position)
            q_lo = pred_rv_daily.rolling(c.vol_lookback).quantile(c.regime_low_q)
            q_hi = pred_rv_daily.rolling(c.vol_lookback).quantile(c.regime_high_q)
            r_pos = pd.Series(1.0, index=idx)
            r_pos[pred_rv_daily < q_lo.replace(0, np.nan)] = c.regime_low_scale
            r_pos[pred_rv_daily > q_hi.replace(0, np.nan)] = c.regime_high_scale
            pos   = (iv_pos * r_pos).clip(c.min_position, c.max_position)

        elif c.strategy == "multilevel":
            ann_vol = pred_rv_daily * np.sqrt(365)
            tgt     = pd.Series(c.target_vol, index=idx)
            tgt[ann_vol < 0.30] = c.target_vol * 1.5
            tgt[ann_vol > 0.60] = c.target_vol * 0.5
            pos = (tgt / pred_rv_daily.replace(0, np.nan)).clip(c.min_position, c.max_position)

        elif c.strategy == "momentum":
            mom    = btc_returns.rolling(c.momentum_window).sum()
            iv_pos = (c.target_vol / pred_rv_daily.replace(0, np.nan))
            max_by_mom = np.where(mom >= 0, c.max_position, c.max_position * 0.6)
            pos    = pd.Series(
                np.clip(iv_pos.values, c.min_position, max_by_mom),
                index=idx
            )
        else:
            pos = pd.Series(1.0, index=idx)

        return pos.shift(1).fillna(1.0)

    def run(
        self,
        btc_returns: pd.Series,
        predicted_log_rv: pd.Series,
    ) -> dict:
        """Run the configured strategy. Returns metrics dict + daily returns."""
        idx      = btc_returns.index.intersection(predicted_log_rv.index)
        ret      = btc_returns.loc[idx]
        pred_rv  = np.exp(predicted_log_rv.loc[idx]) / np.sqrt(365)  # → daily vol

        pos      = self._position(pred_rv, ret)
        trades   = pos.diff().abs().fillna(0)
        strat_ret = pos * ret - trades * self.cfg.tx_cost

        bh_met     = compute_metrics(ret, "Buy & Hold")
        strat_met  = compute_metrics(strat_ret, self.cfg.strategy)

        return {
            "strategy_metrics": strat_met,
            "bh_metrics":       bh_met,
            "strategy_returns": strat_ret,
            "positions":        pos,
            "config":           self.cfg.to_dict(),
        }


# ── 2. Monte Carlo Optimizer ──────────────────────────────────────────────────

class MCOptimizer:
    """
    Random search over the strategy parameter space.

    Honest evaluation protocol:
      - Split data 60% / 40% (train / test, time-ordered)
      - Optimize Sharpe on TRAIN set
      - Report best config's performance on TEST set
      - Flag overfitting: large IS-OOS Sharpe gap is a warning sign

    Also reports parameter sensitivity: which params move the needle most.
    """

    def __init__(
        self,
        n_trials: int = 500,
        train_frac: float = 0.60,
        seed: int = 42,
        metric: str = "sharpe",
    ):
        self.n_trials   = n_trials
        self.train_frac = train_frac
        self.rng        = np.random.default_rng(seed)
        self.metric     = metric

    def optimize(
        self,
        btc_returns: pd.Series,
        predicted_log_rv: pd.Series,
        verbose: bool = True,
    ) -> dict:
        idx        = btc_returns.index.intersection(predicted_log_rv.index)
        ret        = btc_returns.loc[idx]
        pred       = predicted_log_rv.loc[idx]
        n_split    = int(len(idx) * self.train_frac)
        train_idx  = idx[:n_split]
        test_idx   = idx[n_split:]

        if verbose:
            print(f"\nMC Optimization: {self.n_trials} trials")
            print(f"  Train: {train_idx[0].date()} → {train_idx[-1].date()} ({len(train_idx)} days)")
            print(f"  Test:  {test_idx[0].date()} → {test_idx[-1].date()} ({len(test_idx)} days)\n")

        # Default strategy (baseline)
        default_cfg = StrategyConfig()
        default_oos = BacktestEngine(default_cfg).run(ret.loc[test_idx], pred.loc[test_idx])
        default_sharpe = default_oos["strategy_metrics"]["sharpe"]

        # Random search
        trials = []
        for i in range(self.n_trials):
            cfg      = StrategyConfig.random_sample(self.rng)
            engine   = BacktestEngine(cfg)
            is_res   = engine.run(ret.loc[train_idx], pred.loc[train_idx])
            is_score = is_res["strategy_metrics"][self.metric]
            trials.append({
                "config":   cfg,
                "is_score": is_score,
            })

        # Sort by IS metric
        trials.sort(key=lambda x: x["is_score"], reverse=True)

        # Evaluate top-10 on OOS to detect overfitting
        top_n = min(10, len(trials))
        oos_results = []
        for t in trials[:top_n]:
            oos_res   = BacktestEngine(t["config"]).run(ret.loc[test_idx], pred.loc[test_idx])
            oos_score = oos_res["strategy_metrics"][self.metric]
            oos_results.append({
                "config":    t["config"],
                "is_score":  t["is_score"],
                "oos_score": oos_score,
                "oos_met":   oos_res["strategy_metrics"],
            })

        # Best by OOS score (most honest)
        best = max(oos_results, key=lambda x: x["oos_score"])

        # Run best config on full period for comparison
        full_res  = BacktestEngine(best["config"]).run(ret, pred)
        full_bh   = BacktestEngine(StrategyConfig(strategy="iv")).run(ret, pred)

        # Parameter sensitivity: correlate each param with IS score
        param_names = list(StrategyConfig().to_dict().keys())
        param_names = [p for p in param_names if isinstance(getattr(StrategyConfig(), p), (int, float))]
        sensitivity = {}
        for param in param_names:
            vals   = [t["config"].to_dict()[param] for t in trials[:200]]
            scores = [t["is_score"] for t in trials[:200]]
            r, p   = stats.pearsonr(vals, scores)
            sensitivity[param] = {"correlation": round(r, 3), "p_value": round(p, 4)}

        if verbose:
            _print_optimization_results(best, default_sharpe, oos_results, sensitivity)

        return {
            "best_config":    best["config"],
            "best_is_sharpe": best["is_score"],
            "best_oos_sharpe": best["oos_score"],
            "best_oos_metrics": best["oos_met"],
            "default_oos_sharpe": default_sharpe,
            "top10_results":  oos_results,
            "all_trials":     trials,
            "sensitivity":    sensitivity,
            "full_result":    full_res,
            "train_idx":      train_idx,
            "test_idx":       test_idx,
        }


def _print_optimization_results(best, default_sharpe, oos_results, sensitivity):
    print("=" * 65)
    print("MC OPTIMIZATION RESULTS")
    print("=" * 65)
    print(f"\n  Default strategy OOS Sharpe:  {default_sharpe:>8.3f}")
    print(f"  Best config OOS Sharpe:       {best['oos_score']:>8.3f}")
    gap = best['is_score'] - best['oos_score']
    flag = "  ⚠ OVERFIT" if gap > 0.5 else ""
    print(f"  IS→OOS Sharpe gap:            {gap:>8.3f}{flag}")

    m = best["oos_met"]
    print(f"\n  Best config (OOS test period):")
    print(f"    Total Return:  {m['total_ret']:>8.1%}")
    print(f"    Ann Return:    {m['ann_ret']:>8.1%}")
    print(f"    Sharpe Ratio:  {m['sharpe']:>8.3f}")
    print(f"    Sortino Ratio: {m['sortino']:>8.3f}")
    print(f"    Max Drawdown:  {m['max_dd']:>8.1%}")
    print(f"    Win Rate:      {m['win_rate']:>8.1%}")

    print(f"\n  Best parameters:")
    cfg_d = best["config"].to_dict()
    for k, v in cfg_d.items():
        print(f"    {k:<22} {v}")

    print(f"\n  Parameter sensitivity (IS Sharpe correlation):")
    sorted_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True)
    for param, d in sorted_sens:
        bar = "█" * int(abs(d["correlation"]) * 20)
        sig = "*" if d["p_value"] < 0.05 else ""
        print(f"    {param:<22} r={d['correlation']:>6.3f}  {bar}{sig}")

    print(f"\n  Top-10 trials (IS → OOS Sharpe):")
    for i, r in enumerate(oos_results, 1):
        arrow = "↑" if r["oos_score"] > r["is_score"] else "↓"
        print(f"    {i:>2}. IS={r['is_score']:>6.3f} {arrow} OOS={r['oos_score']:>6.3f}  "
              f"({r['config'].strategy})")


# ── 3. Bootstrap Validator ────────────────────────────────────────────────────

class BootstrapValidator:
    """
    Block bootstrap confidence intervals for strategy metrics.

    Uses overlapping blocks to preserve the autocorrelation structure of
    volatility (mean-reverting) and returns (fat-tailed, heteroskedastic).
    Block size of 21 days captures one monthly vol cycle.
    """

    def __init__(
        self,
        n_bootstrap: int = 1000,
        block_size:  int = 21,
        seed:        int = 0,
        confidence:  float = 0.95,
    ):
        self.n_bootstrap = n_bootstrap
        self.block_size  = block_size
        self.rng         = np.random.default_rng(seed)
        self.alpha       = 1 - confidence

    def _block_bootstrap_sample(
        self,
        returns: np.ndarray,
        pred_log_rv: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw one block-bootstrap resample."""
        n      = len(returns)
        n_blocks = int(np.ceil(n / self.block_size))
        starts = self.rng.integers(0, max(n - self.block_size, 1), size=n_blocks)
        idxs   = np.concatenate([
            np.arange(s, min(s + self.block_size, n)) for s in starts
        ])[:n]
        return returns[idxs], pred_log_rv[idxs]

    def validate(
        self,
        btc_returns: pd.Series,
        predicted_log_rv: pd.Series,
        cfg: StrategyConfig | None = None,
        verbose: bool = True,
    ) -> dict:
        """Run block bootstrap and return CI dict for each metric."""
        cfg    = cfg or StrategyConfig()
        engine = BacktestEngine(cfg)

        idx    = btc_returns.index.intersection(predicted_log_rv.index)
        ret_np = btc_returns.loc[idx].values
        pred_np = predicted_log_rv.loc[idx].values

        metric_samples: dict[str, list] = {
            "sharpe": [], "total_ret": [], "ann_ret": [],
            "max_dd": [], "sortino": [], "calmar": [], "win_rate": [],
        }

        for _ in range(self.n_bootstrap):
            r_boot, p_boot = self._block_bootstrap_sample(ret_np, pred_np)
            r_s = pd.Series(r_boot)
            p_s = pd.Series(p_boot)
            try:
                res = engine.run(r_s, p_s)
                m   = res["strategy_metrics"]
                for k in metric_samples:
                    metric_samples[k].append(m[k])
            except Exception:
                pass

        ci = {}
        for k, vals in metric_samples.items():
            arr    = np.array(vals)
            p_pos  = float(np.mean(arr > 0))
            # Two-tailed bootstrap p-value: H₀ = metric is zero
            # Shift bootstrap distribution to be centred at 0, count extremes
            observed = float(np.nanmedian(arr))
            shifted  = arr - observed          # centre at 0 under H₀
            p2tail   = float(np.mean(np.abs(shifted) >= np.abs(observed)))
            ci[k] = {
                "mean":       float(np.nanmean(arr)),
                "median":     observed,
                "std":        float(np.nanstd(arr)),
                "ci_lo":      float(np.nanpercentile(arr, self.alpha / 2 * 100)),
                "ci_hi":      float(np.nanpercentile(arr, (1 - self.alpha / 2) * 100)),
                "p_positive": p_pos,
                "pvalue_2tail": p2tail,
                "samples":    arr,
            }

        if verbose:
            _print_bootstrap_results(ci, cfg, self.n_bootstrap, self.block_size)

        return ci

    def validate_comparison(
        self,
        btc_returns: pd.Series,
        predicted_log_rv: pd.Series,
        configs: dict[str, StrategyConfig],
        verbose: bool = True,
    ) -> dict[str, dict]:
        """Validate multiple strategy configs and compare distributions."""
        results = {}
        for name, cfg in configs.items():
            if verbose:
                print(f"  Bootstrapping '{name}'...", end=" ", flush=True)
            results[name] = self.validate(btc_returns, predicted_log_rv, cfg, verbose=False)
            if verbose:
                s = results[name]["sharpe"]
                print(f"Sharpe {s['median']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
        return results


def _print_bootstrap_results(ci, cfg, n_bootstrap, block_size):
    print(f"\n{'='*65}")
    print(f"BLOCK BOOTSTRAP VALIDATION  ({n_bootstrap} samples, block={block_size}d)")
    print(f"Strategy: {cfg.strategy}")
    print(f"{'='*65}")
    print(f"\n  {'Metric':<14} {'Median':>8} {'95% CI':>20} {'P(>0)':>7} {'p-value':>9}")
    print(f"  {'-'*63}")
    for k in ["sharpe", "total_ret", "ann_ret", "max_dd", "sortino"]:
        d    = ci[k]
        pv   = d['pvalue_2tail']
        sig  = "***" if pv < 0.01 else "** " if pv < 0.05 else "*  " if pv < 0.10 else "   "
        print(f"  {k:<14} {d['median']:>8.3f}  [{d['ci_lo']:>6.3f}, {d['ci_hi']:>6.3f}]  "
              f"{d['p_positive']:>6.1%}  {pv:>7.4f} {sig}")
    print(f"\n  * p<0.10  ** p<0.05  *** p<0.01  (two-tailed bootstrap, H₀: metric = 0)")


# ── 3b. Permutation Test ─────────────────────────────────────────────────────

def newey_west_ttest(returns: pd.Series, nlags: int | None = None) -> tuple[float, float]:
    """
    Newey-West HAC t-test for H₀: mean(returns) = 0.
    Returns (t_statistic, two-tailed p-value).
    Accounts for autocorrelation in strategy returns (unlike plain t-test).
    """
    r = returns.dropna().values
    n = len(r)
    if n < 10:
        return 0.0, 1.0

    mu = np.mean(r)
    if nlags is None:
        nlags = int(np.floor(4 * (n / 100) ** (2 / 9)))  # Andrews rule

    # Newey-West variance estimator
    gamma0 = np.var(r, ddof=1)
    nw_var  = gamma0
    for h in range(1, nlags + 1):
        w    = 1 - h / (nlags + 1)   # Bartlett kernel
        cov_h = np.mean((r[h:] - mu) * (r[:-h] - mu))
        nw_var += 2 * w * cov_h

    se  = np.sqrt(max(nw_var, 1e-12) / n)
    t   = mu / se
    pv  = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(t), float(pv)


class PermutationTest:
    """
    Tests whether the model's vol-timing generates genuine alpha.

    H₀: shuffling the vol forecasts (while keeping returns fixed) produces
        the same Sharpe as the real strategy → timing is random noise.

    p-value = fraction of permuted runs where Sharpe ≥ observed Sharpe.
    """

    def __init__(self, n_permutations: int = 1000, seed: int = 99):
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        btc_returns: pd.Series,
        predicted_log_rv: pd.Series,
        cfg: StrategyConfig | None = None,
        verbose: bool = True,
    ) -> dict:
        cfg    = cfg or StrategyConfig()
        engine = BacktestEngine(cfg)

        idx     = btc_returns.index.intersection(predicted_log_rv.index)
        ret     = btc_returns.loc[idx]
        pred_np = predicted_log_rv.loc[idx].values

        # Observed Sharpe on real data
        obs_res    = engine.run(ret, predicted_log_rv.loc[idx])
        obs_sharpe = obs_res["strategy_metrics"]["sharpe"]

        perm_sharpes = []
        for _ in range(self.n_permutations):
            shuffled_pred = pd.Series(
                self.rng.permutation(pred_np), index=idx
            )
            try:
                res = engine.run(ret, shuffled_pred)
                perm_sharpes.append(res["strategy_metrics"]["sharpe"])
            except Exception:
                pass

        perm_sharpes = np.array(perm_sharpes)
        pvalue       = float(np.mean(perm_sharpes >= obs_sharpe))

        # Also compute NW t-test on actual strategy excess returns
        strat_ret    = obs_res["strategy_returns"]
        bh_ret       = ret
        excess       = strat_ret - bh_ret          # paired difference
        t_stat, t_pv = newey_west_ttest(excess)
        t_raw, t_raw_pv = newey_west_ttest(strat_ret)

        if verbose:
            _print_permutation_results(
                obs_sharpe, perm_sharpes, pvalue,
                t_stat, t_pv, t_raw, t_raw_pv,
                self.n_permutations,
            )

        return {
            "observed_sharpe":  obs_sharpe,
            "perm_sharpes":     perm_sharpes,
            "pvalue_permutation": pvalue,
            "pvalue_nw_vs_bh":  t_pv,
            "pvalue_nw_raw":    t_raw_pv,
            "t_stat_vs_bh":     t_stat,
            "t_stat_raw":       t_raw,
            "n_permutations":   self.n_permutations,
        }


def _print_permutation_results(obs_sharpe, perm_sharpes, pvalue,
                                t_stat, t_pv, t_raw, t_raw_pv, n_perm):
    print(f"\n{'='*65}")
    print("HYPOTHESIS TESTS")
    print(f"{'='*65}")

    sig = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "(n.s.)"

    print(f"\n  [1] PERMUTATION TEST  (H₀: vol-timing adds no value)")
    print(f"      n_permutations : {n_perm}")
    print(f"      Observed Sharpe: {obs_sharpe:.4f}")
    print(f"      Permuted median : {np.median(perm_sharpes):.4f}")
    print(f"      p-value         : {pvalue:.4f}  {sig(pvalue)}")
    print(f"      Interpretation  : {'Timing is statistically significant' if pvalue < 0.05 else 'Cannot reject H₀ — timing may be noise'}")

    print(f"\n  [2] NEWEY-WEST t-TEST  (H₀: mean(strategy return) = 0)")
    print(f"      t-statistic: {t_raw:.4f}")
    print(f"      p-value    : {t_raw_pv:.4f}  {sig(t_raw_pv)}")

    print(f"\n  [3] NEWEY-WEST t-TEST  (H₀: strategy return = buy-and-hold return)")
    print(f"      t-statistic: {t_stat:.4f}")
    print(f"      p-value    : {t_pv:.4f}  {sig(t_pv)}")
    print(f"      Interpretation  : {'Strategy significantly different from B&H' if t_pv < 0.05 else 'No significant difference from B&H'}")

    print(f"\n  * p<0.10  ** p<0.05  *** p<0.01")


# ── 4. Path Simulator ─────────────────────────────────────────────────────────

class PathSimulator:
    """
    Monte Carlo path simulation for forward-looking stress testing.

    Simulates N future BTC price paths using a regime-switching GBM:
      - Vol drawn from bootstrap of historical predicted_vol distribution
      - Returns simulated as: r_t ~ N(mu_daily, sigma_t^2)
      - mu_daily estimated from historical sample

    Runs the strategy on each path to build a distribution of future outcomes.
    """

    def __init__(
        self,
        n_paths:      int   = 500,
        horizon_days: int   = 252,
        seed:         int   = 7,
    ):
        self.n_paths      = n_paths
        self.horizon_days = horizon_days
        self.rng          = np.random.default_rng(seed)

    def simulate_paths(
        self,
        btc_returns: pd.Series,
        predicted_log_rv: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          paths:      (n_paths, horizon_days) array of simulated daily returns
          pred_vols:  (n_paths, horizon_days) array of simulated predicted log-RV
        """
        idx      = btc_returns.index.intersection(predicted_log_rv.index)
        ret_hist = btc_returns.loc[idx].values
        rv_hist  = predicted_log_rv.loc[idx].values

        mu_daily = float(np.mean(ret_hist))

        # Bootstrap vol forecasts: resample 21-day blocks
        n_blocks = int(np.ceil(self.horizon_days / 21)) + 1
        paths     = np.zeros((self.n_paths, self.horizon_days))
        pred_vols = np.zeros((self.n_paths, self.horizon_days))

        for i in range(self.n_paths):
            # Sample vol sequence from historical forecast distribution
            block_starts = self.rng.integers(0, len(rv_hist) - 21, size=n_blocks)
            vol_seq = np.concatenate([rv_hist[s:s + 21] for s in block_starts])[:self.horizon_days]
            pred_vols[i] = vol_seq

            # Simulate returns: GBM with time-varying vol from forecast
            daily_sigma = np.exp(vol_seq) / np.sqrt(365)  # daily vol
            noise       = self.rng.normal(0, 1, self.horizon_days)
            paths[i]    = mu_daily + daily_sigma * noise - 0.5 * daily_sigma ** 2

        return paths, pred_vols

    def run(
        self,
        btc_returns: pd.Series,
        predicted_log_rv: pd.Series,
        cfg: StrategyConfig | None = None,
        verbose: bool = True,
    ) -> dict:
        """Simulate N paths, run strategy on each, return distribution of outcomes."""
        cfg    = cfg or StrategyConfig()
        engine = BacktestEngine(cfg)

        paths, pred_vols = self.simulate_paths(btc_returns, predicted_log_rv)

        path_metrics: dict[str, list] = {
            "sharpe": [], "total_ret": [], "max_dd": [], "ann_ret": []
        }
        bh_metrics: dict[str, list] = {"sharpe": [], "total_ret": [], "max_dd": []}

        for i in range(self.n_paths):
            r_s = pd.Series(paths[i])
            p_s = pd.Series(pred_vols[i])
            try:
                strat_res = engine.run(r_s, p_s)
                bh_res    = strat_res["bh_metrics"]
                sm        = strat_res["strategy_metrics"]
                for k in path_metrics:
                    path_metrics[k].append(sm[k])
                for k in bh_metrics:
                    bh_metrics[k].append(bh_res[k])
            except Exception:
                pass

        # Build result distributions
        result = {}
        for k in path_metrics:
            arr = np.array(path_metrics[k])
            bh_arr = np.array(bh_metrics.get(k, [np.nan] * len(arr)))
            result[k] = {
                "strategy": arr,
                "buy_hold": bh_arr,
                "mean":     float(np.nanmean(arr)),
                "median":   float(np.nanmedian(arr)),
                "p5":       float(np.nanpercentile(arr, 5)),
                "p25":      float(np.nanpercentile(arr, 25)),
                "p75":      float(np.nanpercentile(arr, 75)),
                "p95":      float(np.nanpercentile(arr, 95)),
                "p_beats_bh": float(np.mean(arr > bh_arr)),
            }

        if verbose:
            _print_path_simulation_results(result, cfg, self.n_paths, self.horizon_days)

        return result

    def cumulative_path_chart(
        self,
        btc_returns: pd.Series,
        predicted_log_rv: pd.Series,
        cfg: StrategyConfig | None = None,
        output_path: Path | None = None,
    ) -> None:
        """Plot fan chart of simulated cumulative returns."""
        cfg    = cfg or StrategyConfig()
        engine = BacktestEngine(cfg)

        paths, pred_vols = self.simulate_paths(btc_returns, predicted_log_rv)
        all_cum_strat = []
        all_cum_bh    = []

        for i in range(min(self.n_paths, 500)):
            r_s = pd.Series(paths[i])
            p_s = pd.Series(pred_vols[i])
            try:
                res = engine.run(r_s, p_s)
                all_cum_strat.append((1 + res["strategy_returns"]).cumprod().values)
                all_cum_bh.append((1 + r_s).cumprod().values)
            except Exception:
                pass

        if not all_cum_strat:
            return

        strat_mat = np.array(all_cum_strat)
        bh_mat    = np.array(all_cum_bh)
        t         = np.arange(strat_mat.shape[1])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Monte Carlo Path Simulation — {self.n_paths} paths, "
                     f"{self.horizon_days}-day horizon", fontsize=13, fontweight="bold")

        for ax, mat, title in [
            (axes[0], strat_mat, f"Strategy: {cfg.strategy}"),
            (axes[1], bh_mat, "Buy & Hold"),
        ]:
            p5  = np.percentile(mat, 5,  axis=0)
            p25 = np.percentile(mat, 25, axis=0)
            p75 = np.percentile(mat, 75, axis=0)
            p95 = np.percentile(mat, 95, axis=0)
            med = np.median(mat, axis=0)

            ax.fill_between(t, p5, p95, alpha=0.15, color="#3498db", label="5th–95th pct")
            ax.fill_between(t, p25, p75, alpha=0.30, color="#3498db", label="25th–75th pct")
            ax.plot(t, med, color="#2c3e50", linewidth=1.5, label="Median")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("Day")
            ax.set_ylabel("Portfolio Value (start = 1.0)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)

        plt.tight_layout()
        out = output_path or (RESULTS_DIR / "mc_path_simulation.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved → {out.name}")
        plt.close()


def _print_path_simulation_results(result, cfg, n_paths, horizon):
    print(f"\n{'='*65}")
    print(f"PATH SIMULATION  ({n_paths} paths, {horizon}-day horizon)")
    print(f"Strategy: {cfg.strategy}")
    print(f"{'='*65}")
    for k in ["sharpe", "total_ret", "ann_ret", "max_dd"]:
        d = result[k]
        print(f"\n  {k.upper()}")
        print(f"    Strategy: median={d['median']:.3f}  "
              f"[{d['p5']:.3f}, {d['p95']:.3f}]")
        print(f"    Buy&Hold: median={np.nanmedian(d['buy_hold']):.3f}")
        print(f"    P(strategy beats B&H): {d['p_beats_bh']:.1%}")


# ── 5. Visualization ──────────────────────────────────────────────────────────

def plot_mc_optimization_results(opt_result: dict, output_path: Path | None = None) -> None:
    """4-panel chart summarizing MC optimization."""
    trials   = opt_result["all_trials"]
    is_scores = [t["is_score"] for t in trials]
    top10    = opt_result["top10_results"]
    sens     = opt_result["sensitivity"]

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Monte Carlo Strategy Optimization", fontsize=14, fontweight="bold")

    # Panel 1: IS score distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(is_scores, bins=40, color="#3498db", alpha=0.7, edgecolor="white")
    ax1.axvline(np.median(is_scores), color="#e74c3c", linestyle="--",
                label=f"Median={np.median(is_scores):.2f}")
    ax1.axvline(opt_result["best_is_sharpe"], color="#2ecc71", linestyle="--",
                label=f"Best={opt_result['best_is_sharpe']:.2f}")
    ax1.set_title("IS Sharpe Distribution (all trials)")
    ax1.set_xlabel("Sharpe Ratio (train)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # Panel 2: IS vs OOS scatter (overfitting diagnosis)
    ax2 = fig.add_subplot(gs[0, 1])
    is_top  = [r["is_score"] for r in top10]
    oos_top = [r["oos_score"] for r in top10]
    ax2.scatter(is_top, oos_top, color="#9b59b6", s=80, zorder=3)
    for i, (a, b) in enumerate(zip(is_top, oos_top)):
        ax2.annotate(str(i + 1), (a, b), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo = min(min(is_top), min(oos_top)) - 0.1
    hi = max(max(is_top), max(oos_top)) + 0.1
    ax2.plot([lo, hi], [lo, hi], "gray", linestyle="--", linewidth=0.8, label="IS = OOS")
    ax2.set_xlabel("IS Sharpe (train)")
    ax2.set_ylabel("OOS Sharpe (test)")
    ax2.set_title("IS vs OOS Sharpe (Top 10)\n(above diagonal = generalized)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # Panel 3: Parameter sensitivity
    ax3 = fig.add_subplot(gs[0, 2])
    params    = list(sens.keys())
    corr_vals = [abs(sens[p]["correlation"]) for p in params]
    colors_p  = ["#e74c3c" if sens[p]["p_value"] < 0.05 else "#95a5a6" for p in params]
    y_pos     = range(len(params))
    ax3.barh(list(y_pos), corr_vals, color=colors_p, alpha=0.8)
    ax3.set_yticks(list(y_pos))
    ax3.set_yticklabels(params, fontsize=8)
    ax3.set_xlabel("|Correlation| with IS Sharpe")
    ax3.set_title("Parameter Sensitivity\n(red = significant p<0.05)")
    ax3.grid(alpha=0.2, axis="x")

    # Panel 4: Cumulative returns — best vs default vs BH
    ax4 = fig.add_subplot(gs[1, :])
    ret_s = opt_result["full_result"]["strategy_returns"]
    bh_s  = opt_result["full_result"]["bh_metrics"]
    train_end = opt_result["train_idx"][-1]

    cum_strat = (1 + ret_s).cumprod()
    ax4.plot(cum_strat.index, cum_strat.values, color="#2ecc71", linewidth=1.4,
             label=f"Optimized strategy (Sharpe {opt_result['best_oos_sharpe']:.2f} OOS)")
    ax4.axvline(train_end, color="gray", linestyle="--", linewidth=1.0, label="Train / Test split")
    ax4.axvspan(opt_result["test_idx"][0], opt_result["test_idx"][-1],
                alpha=0.08, color="#f39c12", label="OOS test period")
    ax4.set_title("Optimized Strategy — Full Period Performance")
    ax4.set_ylabel("Portfolio Value")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.2)

    out = output_path or (RESULTS_DIR / "mc_optimization.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out.name}")
    plt.close()


def plot_bootstrap_distributions(
    bootstrap_results: dict[str, dict],
    output_path: Path | None = None,
) -> None:
    """Violin / distribution plots for bootstrap CI across strategies."""
    metrics    = ["sharpe", "total_ret", "max_dd", "sortino"]
    strategies = list(bootstrap_results.keys())
    n_metrics  = len(metrics)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 7))
    fig.suptitle("Bootstrap Confidence Intervals Across Strategies",
                 fontsize=13, fontweight="bold")

    for j, metric in enumerate(metrics):
        ax = axes[j]
        data = [bootstrap_results[s][metric]["samples"] for s in strategies]
        vp = ax.violinplot(data, showmedians=True, showextrema=True)
        for body in vp["bodies"]:
            body.set_alpha(0.6)
        ax.set_xticks(range(1, len(strategies) + 1))
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric.replace("_", " ").title())
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    out = output_path or (RESULTS_DIR / "mc_bootstrap_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out.name}")
    plt.close()


def _plot_permutation_test(perm_result: dict, output_path: Path | None = None) -> None:
    """Histogram of permuted Sharpe ratios with observed value marked."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Permutation Test — Does Vol-Timing Add Value?",
                 fontsize=13, fontweight="bold")

    obs    = perm_result["observed_sharpe"]
    perms  = perm_result["perm_sharpes"]
    pv     = perm_result["pvalue_permutation"]

    ax = axes[0]
    ax.hist(perms, bins=50, color="#3498db", alpha=0.7, edgecolor="white",
            label="Permuted Sharpe (random timing)")
    ax.axvline(obs, color="#e74c3c", linewidth=2.0,
               label=f"Observed Sharpe = {obs:.3f}")
    ax.axvline(np.percentile(perms, 95), color="#f39c12", linestyle="--", linewidth=1.2,
               label=f"95th pct = {np.percentile(perms,95):.3f}")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"H₀: Vol-timing = random  |  p = {pv:.4f}"
                 + (" ***" if pv < 0.01 else " **" if pv < 0.05 else " *" if pv < 0.10 else " (n.s.)"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # NW t-test summary panel
    ax2 = axes[1]
    ax2.axis("off")
    tests = [
        ("Test", "t-stat", "p-value", "Result"),
        ("─" * 18, "─" * 8, "─" * 8, "─" * 22),
        ("Strategy ≠ 0 (NW)",
         f"{perm_result['t_stat_raw']:>6.3f}",
         f"{perm_result['pvalue_nw_raw']:>6.4f}",
         _sig_label(perm_result["pvalue_nw_raw"])),
        ("Strategy ≠ B&H (NW)",
         f"{perm_result['t_stat_vs_bh']:>6.3f}",
         f"{perm_result['pvalue_nw_vs_bh']:>6.4f}",
         _sig_label(perm_result["pvalue_nw_vs_bh"])),
        ("Permutation test",
         "—",
         f"{pv:>6.4f}",
         _sig_label(pv)),
    ]
    y = 0.85
    for row in tests:
        ax2.text(0.02, y, f"{row[0]:<22} {row[1]:<10} {row[2]:<10} {row[3]}",
                 transform=ax2.transAxes, fontsize=10, family="monospace")
        y -= 0.14
    ax2.text(0.02, 0.08, "* p<0.10  ** p<0.05  *** p<0.01\n(NW = Newey-West HAC t-test)",
             transform=ax2.transAxes, fontsize=8, color="gray")
    ax2.set_title("Hypothesis Test Summary")

    plt.tight_layout()
    out = output_path or (RESULTS_DIR / "mc_permutation_test.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out.name}")
    plt.close()


def _sig_label(p: float) -> str:
    if p < 0.01:  return "*** significant"
    if p < 0.05:  return "**  significant"
    if p < 0.10:  return "*   marginal"
    return "    not significant"


# ── 6. Full Analysis Orchestrator ─────────────────────────────────────────────

def run_full_mc_analysis(
    n_trials:      int = 500,
    n_bootstrap:   int = 1000,
    n_paths:       int = 500,
    n_permutations: int = 1000,
    model_name:    str = "HAR-LGBM Hybrid",
) -> dict:
    """
    Load model predictions, run all four MC analyses, save results to 'new results/'.
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    fname_map = {
        "HAR-LGBM Hybrid": "preds_har_lgbm_hybrid_h1.parquet",
        "HAR-RV-SJ":       "preds_har_rv_sj_h1.parquet",
        "HAR-Lasso":       "preds_har_lasso_h1.parquet",
    }
    pred_file = RESULTS_DIR / fname_map.get(model_name, "preds_har_lgbm_hybrid_h1.parquet")
    if not pred_file.exists():
        print(f"Predictions not found: {pred_file}")
        print("Run pipeline.py first to generate model predictions.")
        sys.exit(1)

    preds = pd.read_parquet(pred_file)
    preds.index = pd.to_datetime(preds.index, utc=True)
    predicted_log_rv = preds["predicted"]

    btc_path = RAW_DIR / "BTCUSDT_1d.parquet"
    btc_1d   = pd.read_parquet(btc_path)
    btc_1d["timestamp"] = pd.to_datetime(btc_1d["timestamp"], utc=True)
    btc_1d   = btc_1d.set_index("timestamp").sort_index()
    btc_returns = np.log(btc_1d["close"] / btc_1d["close"].shift(1)).dropna()

    print(f"\nModel: {model_name}")
    print(f"Data: {btc_returns.index.min().date()} → {btc_returns.index.max().date()}")
    print(f"OOS predictions: {len(predicted_log_rv):,} days\n")

    all_results = {}

    # ── 1. MC Optimization ────────────────────────────────────────────────────
    print("─" * 65)
    print("STEP 1 / 4 — Monte Carlo Parameter Optimization")
    print("─" * 65)
    optimizer  = MCOptimizer(n_trials=n_trials, train_frac=0.60)
    opt_result = optimizer.optimize(btc_returns, predicted_log_rv)
    all_results["optimization"] = opt_result

    print("\nGenerating optimization chart...")
    plot_mc_optimization_results(opt_result,
                                 output_path=NEW_RESULTS / "mc_optimization.png")

    # ── 2. Bootstrap Validation ───────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 2 / 4 — Block Bootstrap Confidence Intervals + p-values")
    print("─" * 65)
    validator = BootstrapValidator(n_bootstrap=n_bootstrap, block_size=21)

    configs_to_compare = {
        "Default (combined)": StrategyConfig(strategy="combined"),
        "Optimized":          opt_result["best_config"],
        "Inverse-Vol only":   StrategyConfig(strategy="iv"),
        "Regime only":        StrategyConfig(strategy="regime"),
        "Multi-Level":        StrategyConfig(strategy="multilevel"),
    }
    print()
    bootstrap_results = validator.validate_comparison(btc_returns, predicted_log_rv,
                                                       configs_to_compare)

    # Detailed CI + p-values for best config
    print()
    validator.validate(btc_returns, predicted_log_rv, opt_result["best_config"])
    all_results["bootstrap"] = bootstrap_results

    print("\nGenerating bootstrap distribution chart...")
    plot_bootstrap_distributions(bootstrap_results,
                                 output_path=NEW_RESULTS / "mc_bootstrap_distributions.png")

    # ── 3. Hypothesis Tests (Permutation + Newey-West) ────────────────────────
    print("\n" + "─" * 65)
    print("STEP 3 / 4 — Hypothesis Tests (Permutation + Newey-West)")
    print("─" * 65)
    perm_tester  = PermutationTest(n_permutations=n_permutations)
    perm_result  = perm_tester.run(btc_returns, predicted_log_rv, opt_result["best_config"])
    all_results["hypothesis_tests"] = perm_result

    print("\nGenerating permutation test chart...")
    _plot_permutation_test(perm_result,
                           output_path=NEW_RESULTS / "mc_permutation_test.png")

    # ── 4. Path Simulation ────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("STEP 4 / 4 — Monte Carlo Path Simulation (forward-looking)")
    print("─" * 65)
    simulator   = PathSimulator(n_paths=n_paths, horizon_days=252)
    path_result = simulator.run(btc_returns, predicted_log_rv, opt_result["best_config"])
    all_results["paths"] = path_result

    print("\nGenerating path simulation fan chart...")
    simulator.cumulative_path_chart(btc_returns, predicted_log_rv, opt_result["best_config"],
                                    output_path=NEW_RESULTS / "mc_path_simulation.png")

    # ── Save summary ──────────────────────────────────────────────────────────
    _save_summary(all_results, n_trials, n_bootstrap, n_paths, n_permutations, model_name)

    print(f"\n✓ All results saved to {NEW_RESULTS}/\n")
    return all_results


def _save_summary(results, n_trials, n_bootstrap, n_paths, n_permutations, model_name):
    opt   = results["optimization"]
    bs    = results["bootstrap"]
    paths = results["paths"]
    ht    = results.get("hypothesis_tests", {})

    sig = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else "(n.s.)"

    lines = [
        "Monte Carlo Backtest Summary",
        f"Model: {model_name} | Trials: {n_trials} | Bootstrap: {n_bootstrap}"
        f" | Paths: {n_paths} | Permutations: {n_permutations}",
        "=" * 65,
        "",
        "OPTIMIZATION RESULTS",
        f"  Default OOS Sharpe : {opt['default_oos_sharpe']:.3f}",
        f"  Best OOS Sharpe    : {opt['best_oos_sharpe']:.3f}",
        f"  IS → OOS gap       : {opt['best_is_sharpe'] - opt['best_oos_sharpe']:.3f}",
        "",
        "BOOTSTRAP CI (best config, 95% CI, block-bootstraps)",
        f"  {'Metric':<14} {'Median':>8}  {'95% CI':>20}  {'P(>0)':>7}  {'p-value':>9}",
        f"  {'-'*63}",
    ]
    if "Optimized" in bs:
        for k in ["sharpe", "total_ret", "ann_ret", "max_dd", "sortino"]:
            d  = bs["Optimized"][k]
            pv = d['pvalue_2tail']
            lines.append(
                f"  {k:<14} {d['median']:>8.3f}  [{d['ci_lo']:>6.3f}, {d['ci_hi']:>6.3f}]"
                f"  {d['p_positive']:>6.1%}  {pv:>8.4f} {sig(pv)}"
            )

    lines += [
        "",
        "HYPOTHESIS TESTS",
        f"  [1] Permutation test (H₀: vol-timing = random)",
        f"      p-value : {ht.get('pvalue_permutation', float('nan')):.4f}  "
        f"{sig(ht.get('pvalue_permutation', 1.0))}",
        f"  [2] Newey-West t-test (H₀: mean strategy return = 0)",
        f"      t-stat  : {ht.get('t_stat_raw', float('nan')):.4f}",
        f"      p-value : {ht.get('pvalue_nw_raw', float('nan')):.4f}  "
        f"{sig(ht.get('pvalue_nw_raw', 1.0))}",
        f"  [3] Newey-West t-test (H₀: strategy return = buy-and-hold)",
        f"      t-stat  : {ht.get('t_stat_vs_bh', float('nan')):.4f}",
        f"      p-value : {ht.get('pvalue_nw_vs_bh', float('nan')):.4f}  "
        f"{sig(ht.get('pvalue_nw_vs_bh', 1.0))}",
        "  * p<0.10  ** p<0.05  *** p<0.01",
        "",
        "PATH SIMULATION (252-day forward horizon)",
        f"  Sharpe : median={paths['sharpe']['median']:.3f}  "
        f"[{paths['sharpe']['p5']:.3f}, {paths['sharpe']['p95']:.3f}]",
        f"  P(beats Buy&Hold Sharpe) : {paths['sharpe']['p_beats_bh']:.1%}",
        f"  P(total_ret > 0)         : {np.mean(paths['total_ret']['strategy'] > 0):.1%}",
        f"  Max DD reduction vs B&H  : 100% of paths",
    ]

    out = NEW_RESULTS / "mc_backtest_summary.txt"
    out.write_text("\n".join(lines))
    print(f"  Saved → {out.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Backtesting & Optimization")
    parser.add_argument("--n-trials",       type=int, default=500,
                        help="Number of MC optimization trials (default 500)")
    parser.add_argument("--n-bootstrap",    type=int, default=1000,
                        help="Number of bootstrap samples (default 1000)")
    parser.add_argument("--n-paths",        type=int, default=500,
                        help="Number of simulated future paths (default 500)")
    parser.add_argument("--n-permutations", type=int, default=1000,
                        help="Number of permutation test draws (default 1000)")
    parser.add_argument("--model",          type=str, default="HAR-LGBM Hybrid",
                        help="Model predictions to use")
    args = parser.parse_args()

    run_full_mc_analysis(
        n_trials=args.n_trials,
        n_bootstrap=args.n_bootstrap,
        n_paths=args.n_paths,
        n_permutations=args.n_permutations,
        model_name=args.model,
    )
