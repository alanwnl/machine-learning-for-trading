# %% [markdown]
# # 03 – Normalizing Tick Data: From Noisy ITCH50 Ticks to ML-Ready Bars
# 
# **Learning objective (Stefan Jansen, ML4T 2020)**:  
# Raw tick data suffers from bid-ask bounce and irregular spacing.  
# We convert it into **tick bars**, **time bars**, **volume bars**, and **dollar bars**  
# to reduce noise, stabilize variance, and produce returns that are closer to normal  
# — a prerequisite for almost every linear model, tree-based model, or deep-learning strategy in the book.

# %% [markdown]
# ## Imports & Settings
# %%
import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from math import pi
import plotly.graph_objects as go
from scipy.stats import normaltest

# %%
%matplotlib inline
pd.set_option('display.float_format', lambda x: '%.2f' % x)
sns.set_style('whitegrid')


# %% [markdown]
stock = 'AAPL'
# stock = 'AMZN'
# stock = 'TSLA'
# stock = 'FB'
# %%
# Path configuration – matches the book's data folder structure
data_path = Path('data')
SOURCE_FILE = '10302019.NASDAQ_ITCH50.gz'
itch_store = str(data_path / (Path(SOURCE_FILE).stem + '.h5'))
order_book_store = str(data_path / f"{Path(SOURCE_FILE).stem}_{stock}_order_book.h5")
date = '20191030'
title = '{} | {}'.format(stock, pd.to_datetime(date).date())

# %% [markdown]
# ## Load System Event Data (Market Open/Close)
# 
# ITCH50 messages contain system events (Q = open, M = close).  
# We use them to slice trading hours only — essential for realistic backtesting (see Chapter 8, ML4T Workflow).
# %%
with pd.HDFStore(itch_store) as store:
    sys_events = store['S'].set_index('event_code').drop_duplicates()
    sys_events.timestamp = sys_events.timestamp.add(pd.to_datetime(date)).dt.time
    market_open = sys_events.loc['Q', 'timestamp']
    market_close = sys_events.loc['M', 'timestamp']

# %% [markdown]
# ## Trade Summary – Which Stocks Dominate Volume?
# 
# Quick sanity check: on any given day a few stocks (AAPL, MSFT, etc.) account for most dollar volume.  
# This is why we focus on AAPL for the rest of the notebook.
# %%
with pd.HDFStore(itch_store) as store:
    stocks = store['R'].loc[:, ['stock_locate', 'stock']]
    trades = pd.concat([store['P'], store['Q'].rename(columns={'cross_price': 'price'})], sort=False).merge(stocks)

trades['value'] = trades.shares.mul(trades.price.mul(1e-4))    
trades['value_share'] = trades.value.div(trades.value.sum())

trade_summary_val = trades.groupby('stock').value.sum().sort_values(ascending=False)

fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot absolute value on primary axis
trade_summary_val.iloc[:50].plot.bar(ax=ax1, color='darkblue', title='Traded Value Summary (Top 50)')
ax1.set_ylabel('Total Traded Value ($)')
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '${:,.0f}'.format(y)))

# Add horizontal grid lines for the money axis
ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
ax1.set_axisbelow(True)

# Create proportional secondary axis for percentages
ax2 = ax1.twinx()
total_val = trades.value.sum()
y1_min, y1_max = ax1.get_ylim()
ax2.set_ylim(y1_min / total_val, y1_max / total_val)
ax2.set_ylabel('% of Total Value')
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
ax2.grid(False)

fig.tight_layout()

# %% [markdown]
# ## AAPL Trade Summary – Clean Tick Data
# 
# Load only AAPL trades, convert price from integer (4 decimals) to dollars,  
# remove cross trades, and restrict to regular trading hours.
# %%
with pd.HDFStore(order_book_store) as store:
    trades = store['{}/trades'.format(stock)]

trades.price = trades.price.mul(1e-4)                    # ITCH50 stores price × 10,000
trades = trades[trades.cross == 0]                       # remove opening/closing crosses
trades = trades.between_time(market_open, market_close).drop('cross', axis=1)
trades.info()

# %% [markdown]
# ## Tick Bars – The Rawest View
#
# **Why this matters (Stefan Jansen's key lesson)**:
# - This is the **unprocessed** trade-by-trade price series.
# - Plot reveals extreme noise (bid-ask bounce + irregular timing).
# - Returns are far from normal → violates assumptions of almost every ML algorithm.
# - Goal: visualize the problem before we solve it with time/volume/dollar bars.

# %%
# Create a clean copy so we don't modify the original trades DataFrame
tick_bars = trades.copy()

# IMPORTANT: Change index from full datetime (date + time) to just time-of-day.
# This makes the x-axis show intraday clock time (e.g., 09:30 to 16:00) instead of
# full timestamps, which is much clearer for visual inspection of one trading day.
tick_bars.index = tick_bars.index.time

# Plot the raw price series
tick_bars.price.plot(figsize=(10, 5),
                     title='Tick Bars | {} | {}'.format(stock, pd.to_datetime(date).date()), 
                     lw=1)          # thin line because there are ~100k+ points

plt.xlabel('')                     # remove default label (we already have title)
plt.tight_layout()                 # clean spacing

# %% [markdown]
# ### Reusable Normality Test Function
# 
# Compares any bar type’s returns against the raw tick baseline.  
# Shows clear improvement numbers — exactly what Stefan Jansen wants you to see!

# %%
from scipy.stats import normaltest

def test_normality(returns, name="Data", baseline_stat=None, baseline_p=None):
    """Run D'Agostino-Pearson normality test and print readable comparison."""
    result = normaltest(returns.dropna())
    
    print(f"🔍 Normality Test: {name}")
    print("=" * 50)
    print(f"Test Statistic : {result.statistic:,.2f}")
    print(f"p-value        : {result.pvalue:.4e}")
    print("-" * 50)
    
    if result.pvalue < 0.01:
        print("❌ REJECT normality (p < 0.01) - Not Gaussian, less ideal for ML")
    else:
        print("✅ FAIL TO REJECT normality - Gaussian distribution, good for ML!")
    
    # Comparison with baseline (tick bars)
    if baseline_stat is not None:
        improvement = (baseline_stat - result.statistic) / baseline_stat * 100
        print(f"\n📊 IMPROVEMENT vs Raw Ticks:")
        print(f"   Statistic dropped by {improvement:.1f}% → much better for ML!")
    
    return result

# %%
# Calculate tick returns and establish a baseline for normality
tick_returns = tick_bars.price.pct_change().dropna()
tick_result = test_normality(tick_returns, name="Tick Bars")

tick_stat = tick_result.statistic
tick_p = tick_result.pvalue


# %% [markdown]
# ## Price-Volume Chart Helper Function
# 
# **Purpose (Stefan Jansen's design)**:  
# Create a clean two-panel plot for **any bar type** (time, volume, or dollar bars).  
# Top panel = price (line)  
# Bottom panel = volume (bars)  
# 
# This makes it easy to compare how regularization affects price smoothness and volume flow —  
# a critical visual diagnostic before using the bars in alpha factors or ML models.

# %%
import matplotlib.dates as mdates   # already imported as mpl earlier

def price_volume(df, 
                 price='vwap',           # column to plot on top (usually VWAP)
                 vol='vol',              # column for volume bars
                 suptitle=title,         # main title (e.g. "AAPL | 2019-10-30")
                 fname=None):            # optional: save figure
    
    # 1. Create figure with two vertically stacked subplots (share x-axis)
    fig, axes = plt.subplots(nrows=2, 
                             sharex=True,           # same x-axis for price & volume
                             figsize=(15, 8))       # wide enough for intraday view
    
    # 2. Plot price series (top panel)
    axes[0].plot(df.index, df[price], color='blue', linewidth=1.5)
    
    # 3. Plot volume bars (bottom panel)
    # width = auto-scaled so bars don't overlap even with many bars
    axes[1].bar(df.index, df[vol], 
                width=1/(5*len(df.index)), 
                color='red', alpha=0.7)
    
    # 4. Professional time-axis formatting (intraday only)
    xfmt = mdates.DateFormatter('%H:%M')           # show only hours:minutes
    axes[1].xaxis.set_major_locator(mdates.HourLocator(interval=3))  # tick every 3 hours
    axes[1].xaxis.set_major_formatter(xfmt)
    axes[1].get_xaxis().set_tick_params(which='major', pad=25)  # extra space for labels
    
    # 5. Titles and layout
    axes[0].set_title('Price (VWAP)', fontsize=14)
    axes[1].set_title('Volume', fontsize=14)
    fig.autofmt_xdate()                     # rotate date labels nicely
    fig.suptitle(suptitle, fontsize=16, y=0.98)   # big overall title
    fig.tight_layout()
    plt.subplots_adjust(top=0.90)           # make room for suptitle
    
    # Optional: save the figure
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    
    plt.show()

# %% [markdown]
# ## Time Bars – Chronological Sampling
# 
# **Methodology**: Groups transactions into fixed chronological periods (e.g., 1-minute, 5-minute).
# 
# **Data Processing**: 
# - Collects all tick events that occurred within a strict time interval.
# - Aggregates them to compute the Open, High, Low, Close (OHLC) prices, Volume-Weighted Average Price (VWAP), total volume traded, and the transaction count.
# 
# **Pros/Cons**: 
# - (+) Yields a much smoother price series than raw ticks.
# - (-) Suffers from varying information content. The market open and close are highly active, while midday is quiet. Time bars over-sample quiet periods (creating low-information bars) and under-sample active ones.
# %%
def get_bar_stats(agg_trades):
    """Compute OHLC, VWAP, volume and transaction count for any bar type"""
    vwap = agg_trades.apply(lambda x: np.average(x.price, weights=x.shares)).to_frame('vwap')
    ohlc = agg_trades.price.ohlc()
    vol = agg_trades.shares.sum().to_frame('vol')
    txn = agg_trades.shares.size().to_frame('txn')
    return pd.concat([ohlc, vwap, vol, txn], axis=1)
resampled = trades.groupby(pd.Grouper(freq='1Min'))
time_bars = get_bar_stats(resampled)
# Normality test – already much better than raw ticks
time_result = test_normality(time_bars.vwap.pct_change().dropna(), name="Time Bars (1-Min)", baseline_stat=tick_stat, baseline_p=tick_p)
price_volume(time_bars,
suptitle=f'Time Bars | {stock} | {pd.to_datetime(date).date()}',
fname='time_bars')

# %% [markdown]
# ## Plotly Candlestick (5-min) – Interactive View
# 
# Uses `graph_objects` to render a native inline candlestick chart for interactive visualization of the Time Bars.
# %%
resampled = trades.groupby(pd.Grouper(freq='5Min'))
df = get_bar_stats(resampled)

time5_result = test_normality(df.vwap.pct_change().dropna(), name="Time Bars (5-Min)", baseline_stat=tick_stat, baseline_p=tick_p)

increase = df.close > df.open

fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='#D5E1DD', decreasing_line_color='#F2583E'
)])

fig.update_layout(
    title='AAPL Candlestick (5-min)',
    xaxis_title='Time',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    width=1500,
    height=600,
    plot_bgcolor='white'
)
fig.update_xaxes(showline=True, linewidth=1, linecolor='lightgrey', gridcolor='lightgrey')
fig.update_yaxes(showline=True, linewidth=1, linecolor='lightgrey', gridcolor='lightgrey')
fig.show()

# %% [markdown]
# ## Volume Bars – Information-Driven Sampling
# 
# **Methodology**: Groups transactions together only after a specific cumulative number of *shares* has been traded, rather than based on the clock.
# 
# **Data Processing**:
# - Iterates through consecutive ticks and continuously sums the `shares` traded.
# - Once the cumulative volume threshold is reached (e.g., average volume traded per minute), a new bar is locked in and the OHLC/VWAP stats are generated.
# 
# **Pros/Cons**:
# - (+) Aligns sampling with actual market activity. During earnings or news events, bars are generated rapidly to capture fast changes. During midday lulls, generation slows down.
# - (-) Does not account for the absolute value of the shares traded. A $10,000 tech stock and a $5 penny stock with the same volume threshold represent vastly different economic activity.
# 
# **Q: Why do some Volume Bars still spike significantly above the average threshold?**
# - **A1: Indivisible Block Trades:** If a single massive institutional trade (e.g., 50,000 shares) exceeds the bar threshold (e.g., 10,000 shares), it cannot be divided. All 50,000 shares land in that bar, causing a massive volume spike.
# - **A2: Grouping Overshoot:** The efficient `groupby` rounding algorithm we use groups trades by integer-rounding the threshold division. Rapid surges that heavily overshoot a boundary force all that volume into a single bucket.
# %%
with pd.HDFStore(order_book_store) as store:
    trades = store['{}/trades'.format(stock)]

trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0]
trades = trades.between_time(market_open, market_close).drop('cross', axis=1)

trades_per_min = trades.shares.sum() / (60 * 7.5)          # average shares per minute
trades['cumul_vol'] = trades.shares.cumsum()

df = trades.reset_index()
by_vol = df.groupby(df.cumul_vol.div(trades_per_min).round().astype(int))
vol_bars = pd.concat([by_vol.timestamp.last().to_frame('timestamp'), 
                      get_bar_stats(by_vol)], axis=1)

price_volume(vol_bars.set_index('timestamp'), 
             suptitle=f'Volume Bars | {stock} | {pd.to_datetime(date).date()}')

vol_returns = vol_bars.vwap.pct_change().dropna()
vol_result = test_normality(vol_returns, name="Volume Bars", baseline_stat=tick_stat, baseline_p=tick_p)

# %% [markdown]
# ## Dollar Bars – Economic-Activity Sampling (Recommended!)
# 
# **Methodology**: Similar to volume bars, but groups transactions based on the cumulative *fiat value* exchanged (Price × Shares Traded).
# 
# **Data Processing**:
# - Calculates the dollar value of every transaction.
# - Sums these values sequentially. Once a predetermined dollar threshold is crossed (e.g., average dollar value traded per minute), a new bar is formed.
# 
# **Pros/Cons**:
# - (+) Highly robust to immense price fluctuations (stock splits, multi-year inflation). Best reflects actual market liquidity and information flow.
# - (+) Recovers the most Gaussian (normal) statistical properties of returns, making it the most suitable representation for Machine Learning models!
# 
# **Q: Why do some Dollar Bars still spike significantly above the average threshold?**
# - **A1: Indivisible Block Trades:** If a single massive institutional trade exceeds the dollar threshold we set, it cannot be divided into pieces mathematically. It dumps the entire value into the current bar, creating a spike.
# - **A2: Grouping Overshoot:** Our `groupby` rounding algorithm is optimized for speed. Rapid surges that heavily overshoot the boundary force chunked trades directly into the same rounded integer group.
# %%
with pd.HDFStore(order_book_store) as store:
    trades = store['{}/trades'.format(stock)]

trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0]
trades = trades.between_time(market_open, market_close).drop('cross', axis=1)

value_per_min = trades.shares.mul(trades.price).sum() / (60 * 7.5)
trades['cumul_val'] = trades.shares.mul(trades.price).cumsum()

df = trades.reset_index()
by_value = df.groupby(df.cumul_val.div(value_per_min).round().astype(int))
dollar_bars = pd.concat([by_value.timestamp.last().to_frame('timestamp'), 
                         get_bar_stats(by_value)], axis=1)

price_volume(dollar_bars.set_index('timestamp'), 
             suptitle=f'Dollar Bars | {stock} | {pd.to_datetime(date).date()}')

dollar_returns = dollar_bars.vwap.pct_change().dropna()
dollar_result = test_normality(dollar_returns, name="Dollar Bars", baseline_stat=tick_stat, baseline_p=tick_p)

# %% [markdown]
# ## Normality Summary Table
# %%
summary_df = pd.DataFrame({
    'Bar Type': ['Tick Bars', 'Time Bars (1-Min)', 'Time Bars (5-Min)', 'Volume Bars', 'Dollar Bars'],
    'Test Statistic': [tick_result.statistic, time_result.statistic, time5_result.statistic, vol_result.statistic, dollar_result.statistic],
    'p-value': [tick_result.pvalue, time_result.pvalue, time5_result.pvalue, vol_result.pvalue, dollar_result.pvalue]
})
summary_df.set_index('Bar Type', inplace=True)
summary_df['Statistic % Improvement'] = (tick_result.statistic - summary_df['Test Statistic']) / tick_result.statistic * 100

# Format p-value to scientific notation and add a boolean column for ML readiness
summary_df['p-value (sci)'] = summary_df['p-value'].apply(lambda x: f"{x:.4e}")
summary_df['Is Normal (ML Ready)?'] = summary_df['p-value'] >= 0.01

print("\n" + "="*100)
print("📊 FINAL NORMALITY SUMMARY TABLE")
print("="*100)
# Drop the original unformatted p-value when printing for cleaner display
print(summary_df.drop('p-value', axis=1).to_string())
print("="*100)