# Bitcoin Volatility Forecasting: A Hybrid Econometric-Machine Learning Approach
## Research Presentation Outline

---

## SLIDE 1 — Title Slide

**Title:** Predicting Bitcoin Volatility Using Cross-Asset Spillovers, Seasonality, and Hybrid ML Models

**Subtitle:** A Walk-Forward Evaluation Across 1,530 Days (2022–2026)

**One-liner:** Can we predict how wild Bitcoin will be tomorrow — and trade on it?

---

## SLIDE 2 — Motivation: Why Bitcoin Volatility?

**The problem:**
- Bitcoin's price can move ±10% in a single day — far more than any traditional asset
- Volatility itself is predictable even when price direction is not
- Accurate volatility forecasts have direct applications in:
  - **Risk management** — how much exposure to hold
  - **Options pricing** — volatility is the core input
  - **Portfolio sizing** — scale positions inversely to risk

**The gap in the literature:**
- Most existing models use GARCH or simple HAR — they ignore cross-asset signals
- Few papers incorporate crypto-native features (Fear & Greed, altcoin correlation)
- Almost none combine traditional market spillovers (VIX, S&P 500) with ML methods

**Our question:** Can a hybrid model combining econometric structure with machine learning — fed by both crypto and traditional market signals — outperform the academic benchmark?

---

## SLIDE 3 — What is Volatility? (For non-technical audience)

**Volatility = how much the price moves, not which direction**

- A calm day: BTC moves ±0.5%
- A volatile day: BTC moves ±8%

**Why it matters:**
- If you knew tomorrow would be extremely volatile, you'd hold a smaller position
- If you knew it would be calm, you'd size up

**How we measure it — Realized Volatility (RV):**
- Take Bitcoin's price every 5 minutes (288 data points per day)
- Sum up the squared moves → this gives a model-free, observable measure of actual volatility
- Annualize by ×√365 so it's interpretable (e.g., "48% annualized vol")

---

## SLIDE 4 — Data Sources

| Source | What We Use | Frequency |
|---|---|---|
| **Binance.US** (public API) | BTC, ETH, SOL, BNB price & volume | 5-minute + daily |
| **yfinance** | VIX, S&P 500, Gold, DXY, 10Y Treasury yield | Daily |
| **alternative.me** | Crypto Fear & Greed Index | Daily |

**Coverage:** January 2021 → April 2026 (5+ years)
**Size:** ~553,000 five-minute BTC bars; 1,901 daily feature rows

**Key design choice:** All data is publicly available and free — no paid data vendors required.

---

## SLIDE 5 — Feature Engineering: What Drives Bitcoin Volatility?

We built **54 features** across 5 categories:

**1. Its own history (autoregressive)**
- Yesterday's vol, last week's average, last month's average
- Upside vs. downside realized variance (jumps)
- How much volatility itself has been varying

**2. Crypto cross-asset signals**
- ETH, SOL, BNB volatility — if altcoins spike, BTC follows
- BTC-ETH rolling correlation — high correlation predicts vol

**3. Traditional market spillovers** *(new addition)*
- **VIX** — the "fear index" of equity markets; leads BTC vol by 1-3 days
- **S&P 500** volatility — equity turbulence spills into crypto post-2020
- **DXY** — dollar strength suppresses BTC volatility
- **10Y Treasury yield** — rate shocks tighten risk appetite

**4. Seasonality**
- Day of week, month, time in Bitcoin's 4-year halving cycle
- Weekends are more volatile (thinner liquidity)
- September historically weakest; Q4 historically strongest

**5. Sentiment**
- Crypto Fear & Greed Index — extreme readings predict vol persistence

---

## SLIDE 6 — Why These Features? The Academic Backing

**VIX → BTC volatility** (Bouri et al. 2021, Fang et al. 2022)
- VIX Granger-causes BTC volatility in normal regimes
- Lead time: 1–3 trading days; effect intensifies during market stress

**HAR model structure** (Corsi 2009)
- Long-memory in volatility is best captured with daily/weekly/monthly lags
- Consistently outperforms GARCH in out-of-sample crypto forecasts

**Signed jumps** (Patton & Sheppard 2015)
- Negative jumps (crashes) predict higher future vol than positive jumps
- Asymmetric effect: fear > greed in vol persistence

**Seasonality in crypto** (Baur & Dimpfl 2018, Catania & Grassi 2022)
- Intraday patterns tied to NYSE open and CME futures settlement
- Halving cycle creates measurable regime shifts in realized volatility

---

## SLIDE 7 — The Models

We tested **8 models**, ranging from simple to complex:

| # | Model | Type | Key Idea |
|---|---|---|---|
| 1 | **HAR-RV-SJ** | Econometric | OLS regression on vol lags + jumps. The academic benchmark. |
| 2 | **HAR-Lasso** | Regularized | Adds all 54 features; Lasso zeros out irrelevant ones. |
| 3 | **XGBoost** | ML | Decision trees that learn complex nonlinear patterns. |
| 4 | **LightGBM** | ML | Faster version of XGBoost. |
| 5 | **HAR-LGBM Hybrid** | Hybrid ★ | HAR captures linear structure → LightGBM models what's left over. |
| 6 | **HAR-LGBM Tuned** | Hybrid + HPO | Same hybrid with Optuna-optimized hyperparameters (60 trials). |
| 7 | **HAR-LSTM Hybrid** | Hybrid + DL | HAR captures linear structure → LSTM learns from 22-day sequences. |
| 8 | **Stacking Ensemble** | Meta-model | Ridge regression that learns optimal weights across all base models. |

**Key design principle:** Never let ML replace the econometric structure — use it to learn what the formula gets wrong.

---

## SLIDE 8 — Validation: How We Ensure Honest Results

**The cardinal rule of financial ML: never train on future data.**

We use **sliding walk-forward validation**:

```
Training window (365 days) | Embargo (5 days) | Test (1 day)
        → slide forward one day →
Training window (365 days) | Embargo (5 days) | Test (1 day)
```

- **365-day training window** — rolls forward, always using the most recent year
- **5-day embargo** — gap between train and test prevents leakage from correlated features
- **1,530 test days** — each day was predicted using only information available at that time
- **Refitted every 20 days** — model adapts to regime changes

This mirrors exactly how the model would work in real deployment.

---

## SLIDE 9 — Evaluation Metrics

| Metric | What It Measures | Why We Use It |
|---|---|---|
| **QLIKE** | Primary loss function | Proper scoring rule for variance; robust to extreme vol spikes; standard in academic literature |
| **MSE (log-RV)** | Mean squared error | How far off are predictions on average |
| **MAE (log-RV)** | Mean absolute error | Less sensitive to outliers than MSE |
| **Correlation** | Linear relationship | How well predictions track actual vol over time |
| **Mincer-Zarnowitz R²** | Forecast efficiency | How much of vol variation the model explains |
| **Diebold-Mariano test** | Statistical significance | Tests if one model is significantly better than another |

**QLIKE is our primary metric** — it penalizes underestimating volatility more than overestimating, which is the right asymmetry for risk management applications.

---

## SLIDE 10 — Results: All Models Ranked

*(1,530 out-of-sample test days, Jan 2022 → Apr 2026)*

| Model | QLIKE | MSE(log) | Corr | Beats Baseline? |
|---|---|---|---|---|
| **HAR-LGBM Hybrid** ★ | **0.0444** | **0.0855** | **0.770** | — (best) |
| HAR-LGBM Tuned | 0.0456 | 0.0880 | 0.762 | Yes* |
| HAR-Lasso | 0.0479 | 0.0898 | 0.755 | Yes* |
| Stacking Ensemble | 0.0559 | 0.1036 | 0.710 | Yes* |
| HAR-RV-SJ (baseline) | 0.0575 | 0.1116 | 0.681 | — |
| HAR-LSTM Hybrid | 0.0584 | 0.1157 | 0.668 | No |
| LightGBM | 0.0610 | 0.1236 | 0.644 | No |
| XGBoost | 0.0698 | 0.1494 | 0.585 | No |

**(*) = statistically significant improvement (Diebold-Mariano test, p < 0.05)**

**The hybrid model explains 59% of day-to-day variation in Bitcoin volatility (R² = 0.594)**

---

## SLIDE 11 — Key Finding: Why Hybrid Wins

**Pure ML loses to a 4-variable regression. Why?**

Volatility has strong **linear long-memory** — yesterday's vol is the single best predictor of tomorrow's. The HAR formula captures this efficiently with just 3 coefficients.

When XGBoost or LightGBM tries to learn this from scratch, it wastes model capacity and overfits on the training window.

**The hybrid solution:**
```
Stage 1: HAR-RV-SJ (OLS)       → captures the linear structure perfectly
Stage 2: LightGBM on residuals  → learns what's LEFT: cross-asset effects,
                                   seasonality patterns, VIX spillovers
```

Result: **24% lower QLIKE than the HAR baseline**, statistically significant.
This matches findings in Catania & Sandholdt (2019) and Akyildirim et al. (2021).

---

## SLIDE 12 — Year-by-Year Performance

| Year | Days | Hybrid QLIKE | HAR QLIKE | Improvement | Avg BTC Vol |
|---|---|---|---|---|---|
| 2022 | 338 | 0.0575 | 0.0622 | **+7.6%** | 58.5% |
| 2023 | 365 | 0.0510 | 0.0698 | **+27.0%** | 37.8% |
| 2024 | 366 | 0.0404 | 0.0529 | **+23.8%** | 49.0% |
| 2025 | 365 | 0.0308 | 0.0445 | **+30.7%** | 52.0% |
| 2026 | 96 | 0.0395 | 0.0607 | **+34.9%** | 55.7% |

**The model improves every year** as it accumulates more training data and learns cross-asset relationships more precisely. 2022 was hardest — the FTX collapse made the relationship between traditional and crypto assets unstable.

---

## SLIDE 13 — Traditional Assets: Do They Help?

Adding VIX, S&P 500, Gold, DXY, and 10Y yields:

| Asset | Relationship | When It Matters Most |
|---|---|---|
| **VIX** | + (strongest) | Equity fear spills into BTC 1-3 days later |
| **S&P 500 vol** | + | Most relevant post-2020 as institutional BTC adoption grew |
| **DXY** | − | Dollar strengthening → investors exit risk assets including BTC |
| **10Y yield** | + | Rate shocks → tighter financial conditions → BTC vol spikes |
| **Gold** | Weak/mixed | Safe-haven narrative vs. BTC inconsistent |

**Impact:** Marginal overall improvement (MSE: 0.2862 → 0.2858) but most impactful in **high-volatility regimes** where cross-market fear transmission is strongest.

---

## SLIDE 14 — Backtest: Can We Trade On This?

**Strategy: Use vol forecasts to size BTC positions**

- When predicted vol is **high** → reduce BTC exposure (protect capital)
- When predicted vol is **low** → increase BTC exposure (capture upside)
- Target 2% daily portfolio volatility; cap at 1.5× BTC, floor at 10%

| Strategy | Total Return | Annualized | Sharpe | Max Drawdown |
|---|---|---|---|---|
| **Buy & Hold BTC** | 5.5% | 1.3% | 0.02 | **-71.5%** |
| Inverse-Vol Sizing | 57.2% | 11.4% | 0.27 | -56.3% |
| Vol Regime Filter | 65.4% | 12.8% | 0.28 | -57.0% |
| **Combined Strategy** | **148.0%** | **24.2%** | **0.56** | **-41.7%** |

**The combined strategy turns a near-flat buy-and-hold into a 148% return, cuts the worst drawdown nearly in half, and achieves a Sharpe ratio 28× higher.**

*(All results are out-of-sample; no lookahead bias; 0.1% transaction costs applied)*

---

## SLIDE 15 — Model Limitations & Honest Caveats

**What the model cannot do:**
- Predict *direction* — only magnitude of moves
- Predict black swan events (FTX collapse, geopolitical shocks, exchange hacks)
- The 10 worst forecast days were all sudden external shocks — no model predicted them

**Known limitations:**
- Traditional market features only help on weekdays — weekend values are forward-filled
- LSTM underperformed due to limited training data (~340 sequences per fold)
- Backtest does not account for slippage or market impact
- Model performance could degrade in future structural breaks (new regulations, ETF flows)

**What makes results trustworthy:**
- Strict walk-forward validation — every prediction uses only past information
- 5-day embargo prevents any data leakage
- Statistical significance tested via Diebold-Mariano
- Mincer-Zarnowitz test confirms forecasts are unbiased (α ≈ 0, p = 0.96)

---

## SLIDE 16 — What We Aim to Do Next

**Immediate extensions:**
1. **Add implied volatility** — Deribit DVOL index (BTC options market's own vol forecast)
2. **On-chain data** — active addresses, exchange inflows, miner capitulation (Glassnode)
3. **Funding rates & open interest** — derivatives market sentiment
4. **News/social NLP** — Twitter sentiment scores around major announcements

**Methodological improvements:**
5. **Pre-trained Transformer** — temporal fusion transformer trained on longer history
6. **Regime-switching HAR** — explicitly model bull/bear/halving regimes (MS-HAR)
7. **Multi-horizon forecasting** — extend to h=5 (weekly) and h=22 (monthly)
8. **Probabilistic forecasts** — predict a distribution of outcomes, not just a point estimate

**Application:**
9. **Live paper trading** — deploy the daily forecast in a simulated portfolio
10. **Options strategy** — use vol forecasts to identify mispriced BTC options on Deribit

---

## SLIDE 17 — Summary

**What we built:**
A hybrid volatility forecasting pipeline combining the best of econometrics and machine learning, trained on 5 years of Bitcoin data with 54 features spanning crypto, traditional markets, and sentiment.

**What we found:**
- The HAR-LGBM Hybrid is the best model — 24% improvement over the academic benchmark
- Traditional assets (especially VIX) add predictive power during high-vol regimes
- Pure ML overfits without the econometric backbone
- The model explains 59% of day-to-day variation in BTC volatility

**Why it matters:**
A reliable volatility forecast transforms a near-flat buy-and-hold Bitcoin position (+5.5% over 4 years) into a structured strategy returning 148% with a Sharpe of 0.56 and half the drawdown.

---

## SPEAKING NOTES — Key Points to Emphasize

**On methodology:**
> "The single most important design choice was walk-forward validation. Every prediction I made used only data that would have been available at that exact moment in time. This is what separates a real forecast from a backfit — it's how the model would actually perform if deployed."

**On why hybrid over pure ML:**
> "This is actually a common finding in the academic literature — GARCH and HAR models are surprisingly hard to beat with machine learning alone. The reason is that volatility has a very strong, well-understood linear structure. The right approach is to model that structure econometrically, then use ML to capture what's left over."

**On traditional assets:**
> "VIX is essentially a leading indicator for BTC volatility. When equity markets get scared, that fear bleeds into crypto 1-3 days later. Before 2020 this effect was weak because BTC was isolated. After institutional adoption, the connection became strong enough to be a reliable predictor."

**On the backtest:**
> "The backtest isn't saying 'I can time the Bitcoin market.' It's saying something more modest: when my model says volatility will be high tomorrow, I should hold less Bitcoin — and when it says volatility will be low, I can safely hold more. That risk-parity insight alone generates most of the outperformance."

**On limitations:**
> "The model failed on exactly the days you'd expect it to fail — FTX, the yen carry unwind, the Iran-Israel conflict. These are regime-breaking events. No amount of historical data would have predicted them. The honest answer is: the model is good at forecasting normal volatility dynamics, and no model is good at forecasting black swans."
