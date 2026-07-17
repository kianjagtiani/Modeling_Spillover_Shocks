# Forecasting Bitcoin Volatility from Cross-Asset Spillovers

Spring '26 Cryptocurrency Desk Project \
**Desk Head:** Kian Jagtiani \
**Members on Project:** Kian Jagtiani, Trisha Tjokrosapoetro, Andrew Park

---

Bitcoin's direction is unpredictable; its volatility is not. Can a hybrid model (HAR
econometric structure plus gradient boosting, fed altcoin spillovers, VIX, and
crypto-native sentiment) beat the academic benchmark out of sample, and can you trade
it? We built the full pipeline: 5-minute realized volatility from 2021–2026, 25+ assets
of cross-market features, walk-forward evaluation over 1,552 out-of-sample days, and a
Monte Carlo backtest that takes its own results seriously.

## What we found

**Forecasting: yes.** The HAR-LGBM Hybrid (LightGBM on HAR residuals) beats the
HAR-RV-SJ benchmark on every metric: QLIKE 0.055 vs 0.068, correlation 0.55 vs 0.49, with
a Diebold-Mariano p-value of 5e-14. It is also well calibrated: Mincer-Zarnowitz β =
1.03 against a target of 1.

**Spillovers are real and asymmetric.** Granger causality across the panel: SOL 5-day
volatility leads BTC volatility (p = 0.0006), ETH leads it (p = 0.005), BNB too
(p = 0.035). Altcoin vol predicts Bitcoin vol, not the reverse direction you'd assume
from market-cap ordering.

**Trading it: no, and we can prove it.** An inverse-vol sizing strategy on the forecasts
lands at OOS Sharpe -0.25 (default) to 0.03 (best config), with an in-sample-to-OOS
Sharpe gap of 0.71 that our own summary flags as overfitting. The permutation test says
the vol-timing signal is non-random (p = 0.002), but Newey-West tests can't distinguish
strategy returns from zero (p = 0.26) or from buy-and-hold (p = 0.996). The one robust
benefit: max drawdown improves on buy-and-hold in 100% of simulated forward paths. Good
forecasts, no alpha — the honest headline.

## Conclusion and next steps

Volatility forecasting works; monetizing it through spot position sizing doesn't, because
sizing captures none of the vol premium. The forecast quality points somewhere else:

- Options. `src/options_strategy.py` sketches the Deribit volatility-risk-premium trade
  (IV typically 10–20% over realized). That is where a good RV forecast earns, and it
  needs real options data next.
- The SOL→BTC lead deserves its own study; six days of lead time is a lot.
- The δ-gap overfitting warning means any deployed config needs nested walk-forward, not
  a single grid search.

## Repository contents

- `src/`: the pipeline. Data ingestion, HAR + spillover features, LGBM/XGBoost/LSTM/
  stacking models, walk-forward validation, lead-lag analysis, Monte Carlo backtest with
  permutation and Newey-West tests, options-strategy simulation.
- `btc_volatility_research.ipynb`: the executed end-to-end notebook. `strats.ipynb`:
  15 candidate vol strategies with citations.
- `results/`, `new results/`: forecasts (parquet), scores, statistical summaries, all
  figures, and the presentation.
- `Calendar Effects/`, `code.py`, `code1.py`: the earlier GARCH calendar-effects and
  spillover-alpha experiments this grew out of.
- Prices come from public APIs (see `src/data_ingestion.py`); no keys required.

## References

The HAR-RV framework is Corsi (2009); the semivariance split follows Patton and Sheppard
(2015). The hybrid approach is in the spirit of ML-augmented HAR models in recent
realized-volatility literature, evaluated the way that literature demands: QLIKE loss,
Mincer-Zarnowitz calibration, and Diebold-Mariano tests against the benchmark.
