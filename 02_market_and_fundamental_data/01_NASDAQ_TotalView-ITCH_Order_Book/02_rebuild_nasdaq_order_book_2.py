# %% [markdown]
#   # Working with Order Book Data: NASDAQ ITCH

# %% [markdown]
#   The primary source of market data is the order book, which is continuously updated in real-time throughout the day to reflect all trading activity. Exchanges typically offer this data as a real-time service and may provide some historical data for free.
# 
#   The trading activity is reflected in numerous messages about trade orders sent by market participants. These messages typically conform to the electronic Financial Information eXchange (FIX) communications protocol for real-time exchange of securities transactions and market data or a native exchange protocol.

# %% [markdown]
#   ## Imports

# %%
from pathlib import Path
from collections import Counter
from datetime import timedelta
from datetime import datetime
from time import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# %%
sns.set_style('whitegrid')

# %%
def format_time(t):
    """Return a formatted time string 'HH:MM:SS
    based on a numeric time() value"""
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

# %% [markdown]
#   ### Set Data paths

# %% [markdown]
#   We will store the download in a `data` subdirectory and convert the result to `hdf` format (discussed in the last section of chapter 2).

# %%
data_path = Path('data') # set to e.g. external harddrive
SOURCE_FILE = '10302019.NASDAQ_ITCH50.gz'
itch_store = str(data_path / (Path(SOURCE_FILE).stem + '.h5'))
date = '10302019'
stock = 'AAPL'
order_book_store = data_path / f"{Path(SOURCE_FILE).stem}_{stock}_order_book.h5"

# %% [markdown]
#   ## Build Order Book

# %%
order_dict = {-1: 'sell', 1: 'buy'}

# %% [markdown]
#   The parsed messages allow us to rebuild the order flow for the given day. The 'R' message type contains a listing of all stocks traded during a given day, including information about initial public offerings (IPOs) and trading restrictions.

# %% [markdown]
#   Throughout the day, new orders are added, and orders that are executed and canceled are removed from the order book. The proper accounting for messages that reference orders placed on a prior date would require tracking the order book over multiple days, but we are ignoring this aspect here.

# %% [markdown]
#   ### Get all messages for given stock

# %% [markdown]
#   The `get_messages()` function illustrates how to collect the orders for a single stock that affects trading (refer to the ITCH specification for details about each message):

# %%
def get_messages(date, stock=stock):
    """Collect trading messages for given stock.

    Builds a *live* order registry so that chained U (Replace) messages
    and subsequent E/C/X/D messages always reference the correct
    buy_sell_indicator, current shares, and current price.
    """
    with pd.HDFStore(itch_store) as store:
        # Fixed-format HDF5: load full table, filter in memory (fast with 128GB RAM)
        r_data = store['R']
        stock_locate = r_data[r_data.stock == stock].stock_locate.iloc[0]

        data = {}
        # trading message types
        messages = ['A', 'F', 'E', 'C', 'X', 'D', 'U', 'P', 'Q']
        for m in messages:
            df = store[m]
            data[m] = df[df.stock_locate == stock_locate].drop('stock_locate', axis=1).assign(type=m)

    # -----------------------------------------------------------------
    # Build a LIVE order registry: order_ref -> (buy_sell_indicator, shares, price)
    # This handles chained U messages correctly.
    # -----------------------------------------------------------------
    order_cols = ['order_reference_number', 'buy_sell_indicator', 'shares', 'price']
    initial_orders = pd.concat([data['A'], data['F']], sort=False, ignore_index=True).loc[:, order_cols]

    # Start with A/F orders
    order_registry = {}
    for row in initial_orders.itertuples(index=False):
        order_registry[row.order_reference_number] = (
            row.buy_sell_indicator, row.shares, row.price
        )

    # Process U messages in timestamp order to handle chained replacements
    u_sorted = data['U'].sort_values('timestamp')
    for row in u_sorted.itertuples(index=False):
        orig_ref = row.original_order_reference_number
        new_ref = row.new_order_reference_number
        if orig_ref in order_registry:
            old_buysell, old_shares, old_price = order_registry[orig_ref]
            # Register the NEW order with the new price/shares but same side
            order_registry[new_ref] = (old_buysell, row.shares, row.price)
            # Keep old entry so we can look up the replaced values
        # If orig_ref not found, this is an order from a prior day; skip

    # -----------------------------------------------------------------
    # Now enrich each message type using the registry
    # -----------------------------------------------------------------
    def lookup_order(ref):
        """Return (buy_sell_indicator, shares, price) or (NaN, NaN, NaN)."""
        if ref in order_registry:
            return order_registry[ref]
        return (np.nan, np.nan, np.nan)

    # E, C, X, D: merge with registry on order_reference_number
    for m in ['E', 'C', 'X', 'D']:
        refs = data[m]['order_reference_number']
        looked_up = refs.apply(lookup_order)
        data[m] = data[m].assign(
            buy_sell_indicator=[x[0] for x in looked_up],
            shares=[x[1] for x in looked_up],
            price=[x[2] for x in looked_up]
        )

    # U messages: need both the NEW order info (already in the message)
    # and the REPLACED (old) order info from the registry
    u_data = data['U'].copy()
    old_refs = u_data['original_order_reference_number']
    looked_up_old = old_refs.apply(lookup_order)
    u_data = u_data.assign(
        buy_sell_indicator=[x[0] for x in looked_up_old],
        shares_replaced=[x[1] for x in looked_up_old],
        price_replaced=[x[2] for x in looked_up_old]
    )
    data['U'] = u_data

    # X: use cancelled_shares as shares
    data['X']['shares'] = data['X']['cancelled_shares']
    data['X'] = data['X'].dropna(subset=['price'])

    data['Q'].rename(columns={'cross_price': 'price'}, inplace=True)

    data = pd.concat([data[m] for m in messages], ignore_index=True, sort=False)
    data['date'] = pd.to_datetime(date, format='%m%d%Y')
    data.timestamp = data['date'].add(data.timestamp)
    data = data[data.printable != 0]

    drop_cols = ['tracking_number', 'order_reference_number', 'original_order_reference_number',
                 'cross_type', 'new_order_reference_number', 'attribution', 'match_number',
                 'printable', 'date', 'cancelled_shares']
    return data.drop(drop_cols, axis=1, errors='ignore').sort_values('timestamp').reset_index(drop=True)

# %%
import os
from pathlib import Path

book_exists = False
store_path = Path(order_book_store)
if store_path.exists():
    with pd.HDFStore(store_path, mode='r') as store:
        if f'/{stock}/buy' in store.keys() and f'/{stock}/sell' in store.keys():
            book_exists = True

if book_exists:
    print(f"Order book for {stock} already exists in {order_book_store}. cells will skip rebuild.")

# %%
if not book_exists:
    pass
    print("Loading and filtering messages...")
    SCRIPT_START = time()
    messages = get_messages(date=date)
    messages.info(show_counts=True)

# %%
if not book_exists:
    pass
    with pd.HDFStore(order_book_store) as store:
        key = f'{stock}/messages'
        store.put(key, messages)
        print(store.info())

# %% [markdown]
#   ### Combine Trading Records

# %% [markdown]
#   Reconstructing successful trades, that is, orders that are executed as opposed to those that were canceled from trade-related message types, C, E, P, and Q, is relatively straightforward:

# %%
if not book_exists:
    pass
    def get_trades(m):
        """Combine C, E, P and Q messages into trading records"""
        trade_dict = {'executed_shares': 'shares', 'execution_price': 'price'}
        cols = ['timestamp', 'executed_shares']
        trades = pd.concat([m.loc[m.type == 'E', cols + ['price']].rename(columns=trade_dict),
                            m.loc[m.type == 'C', cols + ['execution_price']].rename(columns=trade_dict),
                            m.loc[m.type == 'P', ['timestamp', 'price', 'shares']],
                            m.loc[m.type == 'Q', ['timestamp', 'price', 'shares']].assign(cross=1),
                            ], sort=False).dropna(subset=['price']).fillna(0)
        return trades.set_index('timestamp').sort_index().astype(int)

# %%
if not book_exists:
    pass
    trades = get_trades(messages)
    print(trades.info())

# %%
if not book_exists:
    pass
    with pd.HDFStore(order_book_store) as store:
        store.put(f'{stock}/trades', trades)

# %% [markdown]
#   ### Create Orders

# %% [markdown]
#   The order book keeps track of limit orders, and the various price levels for buy and sell orders constitute the depth of the order book. To reconstruct the order book for a given level of depth requires the following steps:
# 

# %% [markdown]
#   The `add_orders()` function accumulates sell orders in ascending, and buy orders in descending order for a given timestamp up to the desired level of depth:

# %%
if not book_exists:
    pass
    import bisect

    def add_orders_fast(orders, buysell, nlevels, active_prices):
        """Add orders up to desired depth given by nlevels using pre-sorted price lists.
            sell (buysell=-1) in ascending, buy (buysell=1) in descending order
        """
        new_order = []
        prices = active_prices[buysell]
    
        if buysell == 1:
            # Buy: highest bids first (end of sorted array)
            start_idx = max(0, len(prices) - nlevels)
            for p in reversed(prices[start_idx:]):
                new_order.append((p, orders[p]))
        else:
            # Sell: lowest asks first (start of sorted array)
            end_idx = min(len(prices), nlevels)
            for p in prices[:end_idx]:
                new_order.append((p, orders[p]))
            
        return new_order

# %%
if not book_exists:
    pass
    def save_orders(orders, append=False):
        for buysell, book in orders.items():
            if not book:
                continue
            
            timestamps, prices, shares = [], [], []
            for t, data in book.items():
                for p, s in data:
                    timestamps.append(t)
                    prices.append(p)
                    shares.append(s)
                
            df = pd.DataFrame({'price': prices, 'shares': shares, 'timestamp': timestamps})
            key = f'{stock}/{order_dict[buysell]}'
            df.loc[:, ['price', 'shares']] = df.loc[:, ['price', 'shares']].astype(int)
            with pd.HDFStore(order_book_store) as store:
                if append:
                    store.append(key, df.set_index('timestamp'), format='t')
                else:
                    store.put(key, df.set_index('timestamp'))

# %% [markdown]
#   We iterate over all ITCH messages and process orders and their replacements as required by the specification (this can take a while):

# %%
if not book_exists:
    pass
    order_book = {-1: {}, 1: {}}
    current_orders = {-1: Counter(), 1: Counter()}
    active_prices = {-1: [], 1: []}  # Sorted lists for O(1) access
    message_counter = Counter()
    nlevels = 100

    import os
    import shutil

    # Clean up existing order book data for this stock to prevent duplicate appending
    # We copy all other data to a temporary store, delete the original, and replace it
    # because store.remove() does not reclaim file space in HDF5 and causes file size explosion
    store_path = Path(order_book_store)
    if store_path.exists():
        temp_store_path = store_path.parent / (store_path.stem + "_temp.h5")
    
        with pd.HDFStore(store_path, mode="r") as store:
            keys_to_keep = [k for k in store.keys() if k not in [f"/{stock}/{order_dict[-1]}", f"/{stock}/{order_dict[1]}"]]
        
        if len(keys_to_keep) > 0:
            with pd.HDFStore(store_path, mode="r") as store, \
                 pd.HDFStore(temp_store_path, mode="w", complib="blosc", complevel=9) as new_store:
                for k in keys_to_keep:
                    storer = store.get_storer(k)
                    format_str = "table" if storer.format_type == "table" else "fixed"
                    new_store.put(k, store.get(k), format=format_str)
        
            os.replace(temp_store_path, store_path)
        else:
            # If no keys left, just delete the file entirely
            store_path.unlink()


    start = time()
    for message in messages.itertuples():
        i = message[0]
        if i % 1e5 == 0 and i > 0:
            print(f'{i:,.0f}\t\t{format_time(time() - start)}')
            save_orders(order_book, append=True)
            order_book = {-1: {}, 1: {}}
            start = time()
        if np.isnan(message.buy_sell_indicator):
            continue
        message_counter.update(message.type)

        buysell = message.buy_sell_indicator
        price, shares = None, None

        if message.type in ['A', 'F', 'U']:
            price = int(message.price)
            shares = int(message.shares)

            if price not in current_orders[buysell] or current_orders[buysell][price] == 0:
                bisect.insort(active_prices[buysell], price)
            current_orders[buysell][price] += shares

            new_order = add_orders_fast(current_orders[buysell], buysell, nlevels, active_prices)
            order_book[buysell][message.timestamp] = new_order

        if message.type in ['E', 'C', 'X', 'D', 'U']:
            if message.type == 'U':
                if not np.isnan(message.shares_replaced):
                    price = int(message.price_replaced)
                    shares = -int(message.shares_replaced)
            elif message.type in ['E', 'C']:
                # Bug fix: use executed_shares, not the total order shares
                if not np.isnan(message.price):
                    price = int(message.price)
                    shares = -int(message.executed_shares)
            else:
                # X uses cancelled_shares (set as 'shares' earlier), D uses full order shares
                if not np.isnan(message.price) and not np.isnan(message.shares):
                    price = int(message.price)
                    shares = -int(message.shares)

            if price is not None:
                if price not in current_orders[buysell] or current_orders[buysell][price] == 0:
                    bisect.insort(active_prices[buysell], price)
                current_orders[buysell][price] += shares
            
                if current_orders[buysell][price] <= 0:
                    current_orders[buysell].pop(price, None)
                    # Binary search for removal is fast enough for small N, or we could use a custom remove
                    idx = bisect.bisect_left(active_prices[buysell], price)
                    if idx < len(active_prices[buysell]) and active_prices[buysell][idx] == price:
                        active_prices[buysell].pop(idx)
                    
                new_order = add_orders_fast(current_orders[buysell], buysell, nlevels, active_prices)
                order_book[buysell][message.timestamp] = new_order

    # Save trailing data that didn't hit the 100k boundary
    if order_book[-1] or order_book[1]:
        print(f'Saving final batch...')
        save_orders(order_book, append=True)

    print(f'\nTotal rebuild time: {format_time(time() - SCRIPT_START)}')

# %%
if not book_exists:
    pass
    message_counter = pd.Series(message_counter)
    print(message_counter)

# %%
if not book_exists:
    pass
    with pd.HDFStore(order_book_store) as store:
        print(store.info())

# %% [markdown]
#   ## Order Book Depth

# %%
with pd.HDFStore(order_book_store) as store:
    buy = store[f'{stock}/buy'].reset_index().drop_duplicates()
    sell = store[f'{stock}/sell'].reset_index().drop_duplicates()

# %% [markdown]
#   ### Price to Decimals

# %%
buy.price = buy.price.mul(1e-4)
sell.price = sell.price.mul(1e-4)

# %% [markdown]
#   ### Remove outliers

# %%
percentiles = [.01, .02, .1, .25, .75, .9, .98, .99]
pd.concat([buy.price.describe(percentiles=percentiles).to_frame('buy'),
           sell.price.describe(percentiles=percentiles).to_frame('sell')], axis=1)

# %%
buy = buy[buy.price > buy.price.quantile(.01)]
sell = sell[sell.price < sell.price.quantile(.99)]

# %% [markdown]
#   ### Buy-Sell Order Distribution

# %% [markdown]
#   The number of orders at different price levels, highlighted in the following screenshot using different intensities for buy and sell orders, visualizes the depth of liquidity at any given point in time.

# %% [markdown]
#   The distribution of limit order prices was weighted toward buy orders at higher prices.

# %%
market_open='0930'
market_close = '1600'

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
fig, ax = plt.subplots(figsize=(7,5))
# Use all data within market hours (outliers were removed in previous steps)
buy_filtered = (buy.set_index('timestamp')
                .between_time(market_open, market_close))

sell_filtered = (sell.set_index('timestamp')
                 .between_time(market_open, market_close))

# Modern replacement for the two distplot calls
sns.histplot(data=buy_filtered, x='price', ax=ax, label='Buy',
             alpha=0.5, linewidth=1, bins=50)   # same look as old distplot

sns.histplot(data=sell_filtered, x='price', ax=ax, label='Sell',
             alpha=0.5, linewidth=1, bins=50)

ax.legend(fontsize=10)
ax.set_title('Limit Order Price Distribution')

# Clean formatting (no more set_ticklabels warning!)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${int(x):,}'))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y/1000):,}'))

ax.set_xlabel('Price')
ax.set_ylabel("Shares ('000)")

sns.despine()
fig.tight_layout()
plt.show()

# %% [markdown]
#   ### Order Book Depth

# %%
utc_offset = timedelta(hours=4)
depth = 100

# %%
buy_per_min = (buy
               .groupby([pd.Grouper(key='timestamp', freq='Min'), 'price'])
               .shares
               .sum()
               .apply(np.log)
               .to_frame('shares')
               .reset_index('price')
               .between_time(market_open, market_close)
               .groupby(level='timestamp', as_index=False, group_keys=False)
               .apply(lambda x: x.nlargest(columns='price', n=depth))
               .reset_index())
buy_per_min.timestamp = buy_per_min.timestamp.add(utc_offset).astype(int)
buy_per_min.info()

buy_per_min
# %%
sell_per_min = (sell
                .groupby([pd.Grouper(key='timestamp', freq='Min'), 'price'])
                .shares
                .sum()
                .apply(np.log)
                .to_frame('shares')
                .reset_index('price')
                .between_time(market_open, market_close)
                .groupby(level='timestamp', as_index=False, group_keys=False)
                .apply(lambda x: x.nsmallest(columns='price', n=depth))
                .reset_index())

sell_per_min.timestamp = sell_per_min.timestamp.add(utc_offset).astype(int)
sell_per_min.info()



# %%
with pd.HDFStore(order_book_store) as store:
    trades = store[f'{stock}/trades']
trades.price = trades.price.mul(1e-4)
trades = trades[trades.cross == 0].between_time(market_open, market_close)

trades_per_min = (trades
                  .resample('Min')
                  .agg({'price': 'mean', 'shares': 'sum'}))
trades_per_min.index = trades_per_min.index.to_series().add(utc_offset).astype(int)
trades_per_min.info()

# %% [markdown]
#   The following plots the evolution of limit orders and prices throughout the trading day: the dark line tracks the prices for executed trades during market hours, whereas the red and blue dots indicate individual limit orders on a per-minute basis (see notebook for details)

# %%
sns.set_style('white')
fig, ax = plt.subplots(figsize=(14, 6))

buy_per_min.plot.scatter(x='timestamp',
                         y='price', 
                         c='shares', 
                         ax=ax, 
                         colormap='Blues', 
                         colorbar=False, 
                         alpha=.25)

sell_per_min.plot.scatter(x='timestamp',
                          y='price', 
                          c='shares', 
                          ax=ax, 
                          colormap='Reds', 
                          colorbar=False, 
                          alpha=.25)

title = f'{stock} | {date} | Buy & Sell Limit Order Book | Depth = {depth}'
trades_per_min.price.plot(figsize=(14, 8), 
                          c='k', 
                          ax=ax, 
                          lw=2, 
                          title=title)

xticks = [datetime.fromtimestamp(ts / 1e9).strftime('%H:%M') for ts in ax.get_xticks()]
ax.set_xticklabels(xticks)

ax.set_xlabel('')
ax.set_ylabel('Price', fontsize=12)

red_patch = mpatches.Patch(color='red', label='Sell')
blue_patch = mpatches.Patch(color='royalblue', label='Buy')

plt.legend(handles=[red_patch, blue_patch])
sns.despine()
fig.tight_layout()


