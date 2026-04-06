"""
Volatility Spillover Alpha Backtest
Uses yfinance for both crypto and traditional market data (US-friendly, no geo-blocks).
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# pip install yfinance statsmodels scikit-learn

import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.model_selection import TimeSeriesSplit

# ─────────────────────────────────────────────
# 1. DATA COLLECTION (yfinance — no geo-block)
# ─────────────────────────────────────────────

START = '2020-01-01'

def fetch_yf(ticker, start=START):
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower()
                  for c in df.columns]
    return df

print("Fetching data via yfinance...")
btc   = fetch_yf('BTC-USD')   # Bitcoin
eth   = fetch_yf('ETH-USD')   # Ethereum
sp500 = fetch_yf('^GSPC')     # S&P 500
vix   = fetch_yf('^VIX')      # VIX

print(f"BTC rows: {len(btc)} | SP500 rows: {len(sp500)} | VIX rows: {len(vix)}")

# ─────────────────────────────────────────────
# 2. REALIZED VOLATILITY
# ─────────────────────────────────────────────

def realized_vol(prices, window=7):
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)

btc['rv']   = realized_vol(btc['close'])
eth['rv']   = realized_vol(eth['close'])
sp500['rv'] = realized_vol(sp500['close'])
vix['rv']   = realized_vol(vix['close'])

# ─────────────────────────────────────────────
# 3. ALIGN & MERGE
# ─────────────────────────────────────────────

merged = pd.concat([
    btc['rv'].rename('btc_rv'),
    eth['rv'].rename('eth_rv'),
    sp500['rv'].rename('sp500_rv'),
    vix['rv'].rename('vix_rv'),
    sp500['close'].rename('sp500'),
    vix['close'].rename('vix'),
    btc['close'].rename('btc_close'),
], axis=1, join='inner').dropna()

print(f"\nAligned dataset: {len(merged)} rows | "
      f"{merged.index[0].date()} → {merged.index[-1].date()}")

# ─────────────────────────────────────────────
# 4. GRANGER CAUSALITY
# ─────────────────────────────────────────────

print("\n── Granger Causality: SP500 RV → BTC RV ──")
grangercausalitytests(merged[['btc_rv', 'sp500_rv']], maxlag=5, verbose=True)

print("\n── Granger Causality: VIX RV → BTC RV ──")
grangercausalitytests(merged[['btc_rv', 'vix_rv']], maxlag=5, verbose=True)

# ─────────────────────────────────────────────
# 5. VAR MODEL + FEVD
# ─────────────────────────────────────────────

print("\n── VAR Model ──")
var_data  = merged[['btc_rv', 'sp500_rv', 'vix_rv']].copy()
var_fit   = VAR(var_data).fit(maxlags=5, ic='aic')
print(var_fit.summary())

print("\n── Forecast Error Variance Decomposition (10 steps) ──")
fevd    = var_fit.fevd(10)
fevd_df = pd.DataFrame(fevd.decomp[0], columns=var_data.columns)
fevd_df.index.name = 'horizon'
print("\nBTC RV variance explained by:")
print(fevd_df.tail(5).to_string())

# ─────────────────────────────────────────────
# 6. SIGNAL CONSTRUCTION
# ─────────────────────────────────────────────

merged['vix_z'] = ((merged['vix'] - merged['vix'].rolling(30).mean()) /
                    merged['vix'].rolling(30).std())

merged['signal'] = 0
merged.loc[merged['vix_z'] >  2,  'signal'] = -1   # trad stress → short BTC
merged.loc[merged['vix_z'] < -1,  'signal'] =  1   # trad calm   → long BTC

print(f"\nSignal distribution:\n{merged['signal'].value_counts().to_string()}")

# ─────────────────────────────────────────────
# 7. BACKTEST
# ─────────────────────────────────────────────

TAKER_FEE = 0.001    # 0.1% (Coinbase/Kraken comparable)
SLIPPAGE  = 0.0005   # 0.05%

merged['btc_ret']     = np.log(merged['btc_close'] / merged['btc_close'].shift(1))
merged['sig_lagged']  = merged['signal'].shift(1)
merged['trade']       = merged['sig_lagged'].diff().abs()
merged['cost']        = merged['trade'] * (TAKER_FEE + SLIPPAGE)
merged['strat_ret']   = merged['sig_lagged'] * merged['btc_ret'] - merged['cost']
merged['bnh_ret']     = merged['btc_ret']

# ─────────────────────────────────────────────
# 8. PERFORMANCE METRICS
# ─────────────────────────────────────────────

def performance(returns, label):
    r       = returns.dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std()  * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = (1 + r).cumprod()
    max_dd  = (cum / cum.cummax() - 1).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    win_rt  = (r > 0).mean()
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    print(f"  Ann. Return : {ann_ret*100:>8.2f}%")
    print(f"  Ann. Vol    : {ann_vol*100:>8.2f}%")
    print(f"  Sharpe      : {sharpe:>8.2f}")
    print(f"  Max DD      : {max_dd*100:>8.2f}%")
    print(f"  Calmar      : {calmar:>8.2f}")
    print(f"  Win Rate    : {win_rt*100:>8.2f}%")

clean = merged[['strat_ret', 'bnh_ret']].dropna()
performance(clean['strat_ret'], "Spillover Strategy (net of fees)")
performance(clean['bnh_ret'],   "BTC Buy & Hold")

# ─────────────────────────────────────────────
# 9. WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

print("\n── Walk-Forward Validation (5 folds) ──")
tscv = TimeSeriesSplit(n_splits=5)
fold_sharpes = []

for i, (train_idx, test_idx) in enumerate(tscv.split(merged)):
    train = merged.iloc[train_idx]
    test  = merged.iloc[test_idx].copy()

    vix_mean = train['vix'].mean()
    vix_std  = train['vix'].std()
    test['vix_z_wf'] = (test['vix'] - vix_mean) / vix_std

    test['sig_wf']   = 0
    test.loc[test['vix_z_wf'] >  2, 'sig_wf'] = -1
    test.loc[test['vix_z_wf'] < -1, 'sig_wf'] =  1
    test['strat_wf'] = (test['sig_wf'].shift(1) * test['btc_ret'] -
                        test['sig_wf'].diff().abs() * (TAKER_FEE + SLIPPAGE))

    fold_ret = test['strat_wf'].dropna()
    sharpe   = (fold_ret.mean() / fold_ret.std() * np.sqrt(252)
                if fold_ret.std() > 0 else np.nan)
    fold_sharpes.append(sharpe)
    print(f"  Fold {i+1}: Sharpe = {sharpe:.2f} | "
          f"{test.index[0].date()} → {test.index[-1].date()}")

print(f"\n  Mean OOS Sharpe : {np.nanmean(fold_sharpes):.2f}")
print(f"  Std  OOS Sharpe : {np.nanstd(fold_sharpes):.2f}")

# ─────────────────────────────────────────────
# 10. EXPORT
# ─────────────────────────────────────────────

out = merged[['btc_rv','sp500_rv','vix_rv','vix_z','signal',
              'btc_ret','strat_ret','bnh_ret']].dropna()
out['strat_cum'] = (1 + out['strat_ret']).cumprod()
out['bnh_cum']   = (1 + out['bnh_ret']).cumprod()
out.to_csv('backtest_results.csv')
print("\nResults saved to backtest_results.csv")
print("Done.")