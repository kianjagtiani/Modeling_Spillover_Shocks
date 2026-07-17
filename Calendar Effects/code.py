import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# 1. Load and Prepare Data
df = pd.read_csv('BTC-Daily.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Calculate Log Returns (as defined in Eq. 1 of the paper)
df['Log_Returns'] = np.log(df['close']) - np.log(df['close'].shift(1))
df = df.dropna()

# 2. Extract Calendar Features
df['DayOfWeek'] = df['date'].dt.dayofweek  # 0=Mon, 4=Fri, 6=Sun
df['Month'] = df['date'].dt.month
df['Is_Friday'] = (df['DayOfWeek'] == 4).astype(int)
df['Is_Bullish_Month'] = df['Month'].isin([4, 10]).astype(int)

# 3. Strategy Implementation
# Strategy A: "The Friday Alpha" (Long Fridays, Exit Saturdays)
df['Strategy_Friday'] = np.where(df['DayOfWeek'] == 4, df['Log_Returns'], 0)

# Strategy B: "Seasonal Momentum" (Long April and October only)
df['Strategy_Seasonal'] = np.where(df['Is_Bullish_Month'] == 1, df['Log_Returns'], 0)

# 4. Volatility & Higher Moments (ACD-style analysis)
# The paper suggests crypto returns are non-normal (Eq. 2)
# We use a GARCH model with Skew-t distribution to check Friday's impact on variance
am = arch_model(df['Log_Returns'] * 100, vol='Garch', p=1, q=1, dist='skewt')
res = am.fit(disp='off')

# 5. Performance Metrics
cum_bh = (1 + df['Log_Returns']).cumprod()
cum_friday = (1 + df['Strategy_Friday']).cumprod()
cum_seasonal = (1 + df['Strategy_Seasonal']).cumprod()

# 6. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['date'], cum_bh, label='Buy & Hold', color='gray', alpha=0.5)
plt.plot(df['date'], cum_friday, label='Friday-Only Strategy', color='blue')
plt.plot(df['date'], cum_seasonal, label='April/October Strategy', color='green')
plt.title('Backtesting Calendar Effects from Algieri et al. (2025)')
plt.ylabel('Cumulative Log Return')
plt.legend()
plt.grid(True)
plt.savefig('calendar_backtest.png')

print("Strategy Results:")
print(f"Total B&H Return: {cum_bh.iloc[-1]:.2f}")
print(f"Friday Strategy Return: {cum_friday.iloc[-1]:.2f}")
print(f"April/October Return: {cum_seasonal.iloc[-1]:.2f}")
# Compare mean Friday returns across sub-periods
print("Pre-COVID Friday Mean:", df[df['date'] <= '2019-12-31'][df['DayOfWeek']==4]['Log_Returns'].mean())
print("Post-COVID Friday Mean:", df[df['date'] >= '2021-01-01'][df['DayOfWeek']==4]['Log_Returns'].mean())



# 2. Pre-process dates and sort chronologically
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 3. Calculate Daily Log Returns as defined in the paper: x_t = log(P_t) - log(P_t-1)
# Log returns are preferred for statistical analysis of financial time series
df['Log_Returns'] = np.log(df['close']) - np.log(df['close'].shift(1))
df = df.dropna()

# 4. Filter for Fridays (dayofweek: 0=Mon, 4=Fri, 6=Sun)
df['DayOfWeek'] = df['date'].dt.dayofweek
fridays = df[df['DayOfWeek'] == 4]

# 5. Calculate Means
overall_mean = df['Log_Returns'].mean()
friday_mean = fridays['Log_Returns'].mean()

# 6. Output the Proof
print(f"--- Proof of Friday Alpha ---")
print(f"Overall Daily Mean Log Return: {overall_mean:.5f}")
print(f"Friday Mean Log Return:        {friday_mean:.5f}")
print(f"Ratio (Friday / Overall):      {friday_mean / overall_mean:.2f}x")
