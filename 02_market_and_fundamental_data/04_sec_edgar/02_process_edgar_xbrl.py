# %% [markdown]
# # Process and Analyze SEC EDGAR XBRL Data — Multi-Company Pipeline
#
# This script processes EDGAR (Electronic Data Gathering, Analysis, and Retrieval system) XBRL data for multiple companies and produces
# ML-ready panel datasets with automatic stock split detection.
#
# **Output files:**
# | File | Description |
# |------|-------------|
# | `fundamentals_panel.parquet` | Core features, dense (tags common across ≥50% of companies) |
# | `fundamentals_all_tags.parquet` | ALL tags pivoted wide (~2000+ cols, sparse) |
# | `{ticker}_nums.parquet` | Per-company raw EDGAR data (ALL tags, long format) |
# | `stock_splits.parquet` | All detected splits across companies |
# | `price_data.parquet` | Historical prices for all tickers |



# %%
import warnings
warnings.filterwarnings('ignore')

# %%
from pathlib import Path
import json
import requests
import time

import pandas as pd
import numpy as np
import yfinance as yf

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# %%
sns.set_style('whitegrid')

# %% [markdown]
# ## Configuration

# %%
# === CONFIGURABLE PARAMETERS ===

# Companies to process (add/remove tickers as needed)
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V']

# Minimum fraction of companies a tag must appear in to be included in Tier 3 (core panel)
MIN_COMPANY_COVERAGE = 0.5

# SEC API user agent (required by SEC fair access policy)
SEC_USER_AGENT = 'MachineLearningForTrading_Student admin@ml4t.com'

# Data path
data_path = Path('data')
if not data_path.exists():
    data_path.mkdir()

print(f"📋 Processing {len(TICKERS)} companies: {', '.join(TICKERS)}")
print(f"📁 Data path: {data_path.resolve()}")

# %% [markdown]
# ## Step 1: Ticker → CIK Mapping
#
# Instead of hardcoding company names (fragile), we use SEC's official
# `company_tickers.json` to map ticker symbols to Central Index Keys (CIK).

# %%
def get_ticker_cik_map(tickers, user_agent):
    """
    Fetch CIK mapping from SEC's official company_tickers.json.
    
    Returns:
        dict: {ticker: cik} mapping, e.g. {'AAPL': 320193, 'MSFT': 789019}
    """
    url = 'https://www.sec.gov/files/company_tickers.json'
    headers = {'User-Agent': user_agent}
    
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    
    ticker_set = set(tickers)
    mapping = {}
    
    for entry in data.values():
        if entry['ticker'] in ticker_set:
            mapping[entry['ticker']] = int(entry['cik_str'])
    
    # Report any tickers we couldn't find
    missing = ticker_set - set(mapping.keys())
    if missing:
        print(f"⚠️  Could not find CIK for: {missing}")
    
    return mapping

# %%
print("🔍 Fetching Ticker → CIK mapping from SEC...")
ticker_cik_map = get_ticker_cik_map(TICKERS, SEC_USER_AGENT)

for ticker, cik in sorted(ticker_cik_map.items()):
    print(f"  {ticker}: CIK {cik}")

print(f"\n✅ Mapped {len(ticker_cik_map)}/{len(TICKERS)} tickers")

# %% [markdown]
# ## Step 2: Extract Filings for All Companies
#
# Scan all quarterly `sub.parquet` files to find 10-Q and 10-K filings 
# for each company by CIK.

# %%
def extract_company_filings(data_path, ticker_cik_map):
    """
    Extract all 10-Q and 10-K filings for each company across all quarters.
    
    Returns:
        dict: {ticker: DataFrame of submissions}
    """
    # Build reverse map: cik -> ticker
    cik_to_ticker = {cik: ticker for ticker, cik in ticker_cik_map.items()}
    target_ciks = set(cik_to_ticker.keys())
    
    all_subs = []
    sub_files = sorted(data_path.glob('**/sub.parquet'))
    
    print(f"📂 Scanning {len(sub_files)} quarterly sub.parquet files...")
    
    for sub_file in sub_files:
        try:
            sub_data = pd.read_parquet(sub_file)
            # Filter to our target companies and relevant form types
            mask = (sub_data.cik.astype(int).isin(target_ciks)) & \
                   (sub_data.form.isin(['10-Q', '10-K']))
            matches = sub_data[mask].copy()
            if not matches.empty:
                matches['_source_file'] = str(sub_file)
                all_subs.append(matches)
        except Exception as e:
            print(f"  ⚠️  Error reading {sub_file}: {e}")
    
    if not all_subs:
        print("❌ No filings found!")
        return {}
    
    combined = pd.concat(all_subs, ignore_index=True)
    combined['ticker'] = combined.cik.astype(int).map(cik_to_ticker)
    
    # Group by ticker
    filings_by_ticker = {}
    for ticker in ticker_cik_map:
        ticker_filings = combined[combined.ticker == ticker]
        if not ticker_filings.empty:
            # Deduplicate by accession number
            ticker_filings = ticker_filings.drop_duplicates(subset='adsh')
            filings_by_ticker[ticker] = ticker_filings
            n_10q = (ticker_filings.form == '10-Q').sum()
            n_10k = (ticker_filings.form == '10-K').sum()
            print(f"  {ticker}: {n_10q} × 10-Q, {n_10k} × 10-K")
        else:
            print(f"  {ticker}: ❌ No filings found")
    
    return filings_by_ticker

# %%
filings_by_ticker = extract_company_filings(data_path, ticker_cik_map)
print(f"\n✅ Found filings for {len(filings_by_ticker)}/{len(TICKERS)} companies")

# %% [markdown]
# ## Step 3: Extract Numerical Data for All Companies
#
# For each company, collect all numerical data points from their filings
# across all quarters. Save per-ticker raw data (Tier 1).

# %%
def extract_numerical_data(data_path, filings_by_ticker):
    """
    Extract all numerical data for each company's filings.
    Saves per-ticker raw data as {ticker}_nums.parquet (Tier 1).
    
    Returns:
        dict: {ticker: DataFrame of numerical data}
    """
    # Collect all accession numbers we need
    all_adshs = set()
    adsh_to_ticker = {}
    for ticker, filings in filings_by_ticker.items():
        for adsh in filings.adsh.unique():
            all_adshs.add(adsh)
            adsh_to_ticker[adsh] = ticker
    
    print(f"🔢 Extracting numerical data for {len(all_adshs)} filings...")
    
    # Scan all num.parquet files
    all_nums = []
    num_files = sorted(data_path.glob('**/num.parquet'))
    
    for num_file in num_files:
        try:
            num_data = pd.read_parquet(num_file).drop('dimh', axis=1, errors='ignore')
            matches = num_data[num_data.adsh.isin(all_adshs)]
            if not matches.empty:
                all_nums.append(matches)
        except Exception as e:
            print(f"  ⚠️  Error reading {num_file}: {e}")
    
    if not all_nums:
        print("❌ No numerical data found!")
        return {}
    
    combined = pd.concat(all_nums, ignore_index=True)
    combined['ticker'] = combined.adsh.map(adsh_to_ticker)
    combined['ddate'] = pd.to_datetime(combined['ddate'], format='%Y%m%d')
    
    # Split by ticker and save Tier 1 files
    nums_by_ticker = {}
    for ticker in filings_by_ticker:
        ticker_nums = combined[combined.ticker == ticker].copy()
        if not ticker_nums.empty:
            nums_by_ticker[ticker] = ticker_nums
            
            # Save Tier 1: per-company raw data
            output_file = data_path / f'{ticker}_nums.parquet'
            ticker_nums.to_parquet(output_file, index=False)
            
            n_tags = ticker_nums.tag.nunique()
            n_rows = len(ticker_nums)
            print(f"  {ticker}: {n_rows:,} data points, {n_tags} unique tags → {output_file.name}")
        else:
            print(f"  {ticker}: ❌ No numerical data found")
    
    return nums_by_ticker

# %%
nums_by_ticker = extract_numerical_data(data_path, filings_by_ticker)
print(f"\n✅ Extracted numerical data for {len(nums_by_ticker)} companies")

# %% [markdown]
# ## Step 4: Detect Stock Splits
#
# Uses dual strategy:
# 1. **Primary:** `yfinance` `ticker.splits` — reliable and comprehensive
# 2. **Secondary:** EDGAR XBRL tags — cross-validation from filing data
#
# No more hardcoded `stock_split = 7`!

# %%
def detect_stock_splits(tickers, nums_by_ticker):
    """
    Detect stock splits using yfinance (primary) and EDGAR XBRL tags (secondary).
    
    Returns:
        DataFrame with columns: ticker, split_date, split_ratio, source
    """
    splits_records = []
    
    for ticker in tickers:
        # --- Source 1: yfinance (primary, reliable) ---
        try:
            yf_ticker = yf.Ticker(ticker)
            yf_splits = yf_ticker.splits
            if yf_splits is not None and len(yf_splits) > 0:
                for date_val, ratio in yf_splits.items():
                    if ratio != 0 and ratio != 1.0:  # Filter out non-splits
                        split_date = pd.Timestamp(date_val).tz_localize(None)
                        splits_records.append({
                            'ticker': ticker,
                            'split_date': split_date,
                            'split_ratio': float(ratio),
                            'source': 'yfinance'
                        })
            time.sleep(0.2)  # Rate limiting
        except Exception as e:
            print(f"  ⚠️  yfinance split lookup failed for {ticker}: {e}")
        
        # --- Source 2: EDGAR XBRL tags (secondary) ---
        if ticker in nums_by_ticker:
            nums = nums_by_ticker[ticker]
            # Look for stock split related tags
            split_mask = nums.tag.str.contains(
                'StockSplit|SplitConversion|StockholdersEquityNoteStockSplit',
                case=False, na=False
            )
            split_data = nums[split_mask]
            if not split_data.empty:
                for _, row in split_data.iterrows():
                    splits_records.append({
                        'ticker': ticker,
                        'split_date': row['ddate'],
                        'split_ratio': float(row['value']),
                        'source': f'edgar_xbrl:{row["tag"]}'
                    })
    
    if not splits_records:
        print("  No stock splits detected")
        return pd.DataFrame(columns=['ticker', 'split_date', 'split_ratio', 'source'])
    
    splits_df = pd.DataFrame(splits_records)
    splits_df['split_date'] = pd.to_datetime(splits_df['split_date'])
    
    return splits_df

# %%
print("🔄 Detecting stock splits...")
splits_df = detect_stock_splits(TICKERS, nums_by_ticker)

if not splits_df.empty:
    # Show results grouped by source
    print("\n📊 Detected Stock Splits:")
    print("-" * 70)
    for ticker in splits_df.ticker.unique():
        ticker_splits = splits_df[splits_df.ticker == ticker]
        yf_splits = ticker_splits[ticker_splits.source == 'yfinance']
        edgar_splits = ticker_splits[ticker_splits.source != 'yfinance']
        
        if not yf_splits.empty:
            for _, row in yf_splits.iterrows():
                print(f"  {row.ticker}: {row.split_date.strftime('%Y-%m-%d')} "
                      f"ratio={row.split_ratio} (yfinance)")
        if not edgar_splits.empty:
            for _, row in edgar_splits.iterrows():
                print(f"  {row.ticker}: {row.split_date.strftime('%Y-%m-%d')} "
                      f"ratio={row.split_ratio} ({row.source})")
    
    # Save Tier: stock splits lookup table
    splits_df.to_parquet(data_path / 'stock_splits.parquet', index=False)
    print(f"\n✅ Saved {len(splits_df)} split records → stock_splits.parquet")
else:
    print("  No splits found in the data range")

# %% [markdown]
# ## Step 5: Compute Split Adjustment Factors
#
# For each (ticker, date), compute a cumulative split adjustment factor
# so that all historical per-share data is comparable to current shares.

# %%
def compute_split_adj_factors(splits_df, tickers):
    """
    Compute cumulative split adjustment factors for each ticker.
    Uses yfinance splits as the authoritative source.
    
    The factor converts old per-share values to current-share-equivalent:
        adjusted_value = original_value / cumulative_factor
    
    Returns:
        dict: {ticker: Series indexed by split_date with cumulative adjustment factors}
    """
    adj_factors = {}
    
    # Use only yfinance splits (primary, reliable)
    yf_splits = splits_df[splits_df.source == 'yfinance'].copy()
    
    for ticker in tickers:
        ticker_splits = yf_splits[yf_splits.ticker == ticker].sort_values('split_date')
        if ticker_splits.empty:
            adj_factors[ticker] = pd.Series(dtype=float)
            continue
        
        # Cumulative product of split ratios from newest to oldest
        # e.g., AAPL: 7:1 (2014) then 4:1 (2020) → factor before 2014 = 28
        factors = ticker_splits.set_index('split_date')['split_ratio']
        # Reverse cumulative product: for any date, multiply all future split ratios
        cum_factors = factors.sort_index(ascending=False).cumprod().sort_index()
        adj_factors[ticker] = cum_factors
    
    return adj_factors

# %%
split_adj = compute_split_adj_factors(splits_df, TICKERS)

for ticker, factors in split_adj.items():
    if not factors.empty:
        print(f"  {ticker}: {len(factors)} split(s), cumulative factors: "
              f"{dict(zip(factors.index.strftime('%Y-%m-%d'), factors.values))}")

# %% [markdown]
# ## Step 6: Build Fundamental Features per Company
#
# For each company, extract the most recent value of each tag per filing period,
# apply split adjustments to per-share metrics, and compute derived features.

# %%
# Per-share tags that need split adjustment
PER_SHARE_TAGS = {
    'EarningsPerShareDiluted',
    'EarningsPerShareBasic',
    'EarningsPerShareBasicAndDiluted',
    'CommonStockDividendsPerShareDeclared',
    'CommonStockDividendsPerShareCashPaid',
}

def get_split_factor_for_date(split_factors, date):
    """Get the split adjustment factor for a given date."""
    if split_factors.empty:
        return 1.0
    # Find all splits that happened AFTER this date
    future_splits = split_factors[split_factors.index > date]
    if future_splits.empty:
        return 1.0
    # Product of all future split ratios
    return future_splits.prod()


def build_company_features(ticker, nums, split_factors):
    """
    Build a (date → tag) feature matrix for one company.
    
    Strategy:
    - Determine the primary reporting period end date for each filing (adsh)
    - Filter out historical comparative data older than the primary date
    - Snap all tags in the filing to the primary date for perfect alignment
    - Prefer quarterly observations (qtrs == 1) where available
    - Apply split adjustments to per-share metrics
    
    Returns:
        DataFrame with index=ddate, columns=tag names, values=adjusted values
    """
    if nums.empty:
        return pd.DataFrame()
    
    # --- CLEANUP: Align Dates & Drop Historicals ---
    # 1. Determine primary reporting date per filing (max ddate)
    adsh_primary = nums.groupby('adsh')['ddate'].max().rename('primary_ddate')
    nums = nums.merge(adsh_primary, on='adsh')
    
    # 2. Filter out data points older than the primary date 
    # (Allowing a 5-day window for occasional weekend/leap-year tagging artifacts)
    date_diff = (nums['primary_ddate'] - nums['ddate']).dt.days
    nums_current = nums[date_diff.abs() <= 5].copy()
    
    records = []
    
    for tag in nums_current.tag.unique():
        tag_data = nums_current[nums_current.tag == tag].copy()
        
        # Prefer quarterly (qtrs==1) observations for flow statements
        if 'qtrs' in tag_data.columns:
            quarterly = tag_data[tag_data.qtrs == 1]
            if not quarterly.empty:
                tag_data = quarterly
        
        # Keep most recent data point per filing within the current period
        if not tag_data.empty:
            latest = tag_data.groupby('adsh').apply(
                lambda x: x.nlargest(1, 'ddate'), include_groups=False
            ).reset_index(drop=True)
            
            for _, row in latest.iterrows():
                value = row['value']
                # Snap to primary reporting date to guarantee alignment
                final_date = row['primary_ddate']
                
                # Apply split adjustment for per-share metrics
                if tag in PER_SHARE_TAGS and not split_factors.empty:
                    factor = get_split_factor_for_date(split_factors, final_date)
                    if factor > 1:
                        value = value / factor
                
                records.append({
                    'ddate': final_date,
                    'tag': tag,
                    'value': value
                })
    
    if not records:
        return pd.DataFrame()
    
    records_df = pd.DataFrame(records)
    
    # Pivot: rows=dates, columns=tags
    # If duplicates exist across different filings for the same snapped date, take the last
    pivoted = records_df.pivot_table(
        index='ddate', columns='tag', values='value', aggfunc='last'
    )
    
    return pivoted

# %%
print("🏗️  Building fundamental features per company...")

features_by_ticker = {}
for ticker in TICKERS:
    if ticker not in nums_by_ticker:
        print(f"  {ticker}: ⏭️  skipped (no data)")
        continue
    
    sf = split_adj.get(ticker, pd.Series(dtype=float))
    features = build_company_features(ticker, nums_by_ticker[ticker], sf)
    
    if not features.empty:
        features_by_ticker[ticker] = features
        print(f"  {ticker}: {features.shape[0]} dates × {features.shape[1]} tags")
    else:
        print(f"  {ticker}: ❌ no features extracted")

print(f"\n✅ Built features for {len(features_by_ticker)} companies")

# %% [markdown]
# ## Step 7: Download Price Data
#
# Download historical prices for all tickers using yfinance.

# %%
def download_price_data(tickers, start_date='2009-01-01'):
    """
    Download historical adjusted close prices for all tickers.
    
    Returns:
        DataFrame with MultiIndex (ticker, date) and price columns
    """
    print(f"📈 Downloading price data for {len(tickers)} tickers...")
    
    all_prices = []
    
    for ticker in tickers:
        try:
            stock = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
            if stock.empty:
                print(f"  {ticker}: ⚠️  no price data")
                continue
            
            stock.index = pd.to_datetime(stock.index).tz_localize(None)
            
            # Flatten MultiIndex columns if present
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.get_level_values(0)
            
            stock['ticker'] = ticker
            stock = stock.reset_index()
            stock = stock.rename(columns={'Date': 'date', 'index': 'date'})
            
            # Ensure 'date' column exists
            if 'date' not in stock.columns:
                date_cols = [c for c in stock.columns if 'date' in c.lower() or 'Date' in c]
                if date_cols:
                    stock = stock.rename(columns={date_cols[0]: 'date'})
                else:
                    stock['date'] = stock.index
            
            all_prices.append(stock)
            print(f"  {ticker}: {len(stock):,} trading days")
            time.sleep(0.2)
        except Exception as e:
            print(f"  {ticker}: ❌ failed: {e}")
    
    if not all_prices:
        return pd.DataFrame()
    
    prices = pd.concat(all_prices, ignore_index=True)
    
    # Standardize column names
    col_rename = {}
    for col in prices.columns:
        if col.lower() in ('adj close', 'adjclose', 'adj_close'):
            col_rename[col] = 'adj_close'
        elif col.lower() == 'close' and 'adj_close' not in prices.columns:
            col_rename[col] = 'close'
        elif col.lower() == 'open':
            col_rename[col] = 'open'
        elif col.lower() == 'high':
            col_rename[col] = 'high'
        elif col.lower() == 'low':
            col_rename[col] = 'low'
        elif col.lower() == 'volume':
            col_rename[col] = 'volume'
    prices = prices.rename(columns=col_rename)
    
    return prices

# %%
price_df = download_price_data(list(features_by_ticker.keys()))

if not price_df.empty:
    price_df.to_parquet(data_path / 'price_data.parquet', index=False)
    print(f"\n✅ Saved price data → price_data.parquet ({len(price_df):,} rows)")

# %% [markdown]
# ## Step 8: Assemble ML-Ready Panel DataFrames
#
# **Tier 2:** ALL tags pivoted into `fundamentals_all_tags.parquet`
# **Tier 3:** Core features (tags common across ≥50% of companies) into `fundamentals_panel.parquet`
# Both use MultiIndex (ticker, date).

# %%
def assemble_panel(features_by_ticker, price_df, min_coverage=0.5):
    """
    Assemble the ML-ready panel DataFrames.
    
    Returns:
        (tier2_all_tags, tier3_core): Both DataFrames with MultiIndex (ticker, date)
    """
    # --- Combine all company features ---
    all_panels = []
    for ticker, features in features_by_ticker.items():
        df = features.copy()
        df['ticker'] = ticker
        df.index.name = 'date'
        df = df.reset_index()
        all_panels.append(df)
    
    if not all_panels:
        return pd.DataFrame(), pd.DataFrame()
    
    combined = pd.concat(all_panels, ignore_index=True)
    
    # --- Set MultiIndex ---
    combined = combined.set_index(['ticker', 'date']).sort_index()
    
    # --- Tier 2: ALL tags ---
    tier2 = combined.copy()
    
    # Add price columns if available
    if not price_df.empty:
        price_cols = ['close', 'adj_close', 'open', 'high', 'low', 'volume']
        available_price_cols = [c for c in price_cols if c in price_df.columns]
        
        if available_price_cols:
            # Resample prices to match fundamental dates
            price_pivot = price_df.set_index(['ticker', 'date'])[available_price_cols]
            
            # For each (ticker, fundamental_date), get the closest prior price
            for ticker in features_by_ticker:
                ticker_prices = price_df[price_df.ticker == ticker].set_index('date')[available_price_cols]
                if ticker_prices.empty:
                    continue
                
                ticker_dates = combined.loc[ticker].index
                for dt in ticker_dates:
                    # Find closest price on or before this date
                    prior_prices = ticker_prices[ticker_prices.index <= dt]
                    if not prior_prices.empty:
                        closest_price = prior_prices.iloc[-1]
                        for col in available_price_cols:
                            tier2.loc[(ticker, dt), f'price_{col}'] = closest_price[col]
    
    # --- Tier 3: Core features (tags present in >= min_coverage of companies) ---
    n_companies = len(features_by_ticker)
    tag_coverage = combined.notna().groupby(level=0).any().sum() / n_companies
    core_tags = tag_coverage[tag_coverage >= min_coverage].index.tolist()
    
    # Also include any price columns we added
    price_columns = [c for c in tier2.columns if c.startswith('price_')]
    core_columns = list(set(core_tags + price_columns))
    core_columns = [c for c in core_columns if c in tier2.columns]
    
    tier3 = tier2[core_columns].copy()
    
    # Add derived features to Tier 3
    if 'EarningsPerShareDiluted' in tier3.columns and 'price_close' in tier3.columns:
        tier3['pe_ratio'] = tier3['price_close'] / tier3['EarningsPerShareDiluted']
        # Cap extreme P/E values
        tier3.loc[tier3['pe_ratio'].abs() > 500, 'pe_ratio'] = np.nan
    
    if 'NetIncomeLoss' in tier3.columns and 'Revenues' in tier3.columns:
        tier3['net_margin'] = tier3['NetIncomeLoss'] / tier3['Revenues']
    
    if 'GrossProfit' in tier3.columns and 'Revenues' in tier3.columns:
        tier3['gross_margin'] = tier3['GrossProfit'] / tier3['Revenues']
    
    return tier2, tier3

# %%
print("🧩 Assembling ML-ready panel DataFrames...")

tier2_all, tier3_core = assemble_panel(
    features_by_ticker, price_df, min_coverage=MIN_COMPANY_COVERAGE
)

if not tier2_all.empty:
    print(f"\n📊 Tier 2 (ALL tags): {tier2_all.shape[0]} rows × {tier2_all.shape[1]} columns")
    print(f"   Sparsity: {tier2_all.isna().mean().mean():.1%}")
    tier2_all.to_parquet(data_path / 'fundamentals_all_tags.parquet')
    print(f"   ✅ Saved → fundamentals_all_tags.parquet")

if not tier3_core.empty:
    print(f"\n📊 Tier 3 (Core features): {tier3_core.shape[0]} rows × {tier3_core.shape[1]} columns")
    print(f"   Sparsity: {tier3_core.isna().mean().mean():.1%}")
    tier3_core.to_parquet(data_path / 'fundamentals_panel.parquet')
    print(f"   ✅ Saved → fundamentals_panel.parquet")
    
    print(f"\n   Core columns ({len(tier3_core.columns)}):")
    for col in sorted(tier3_core.columns):
        coverage = tier3_core[col].notna().mean()
        print(f"     {col}: {coverage:.0%} non-null")

# %% [markdown]
# ## Step 9: Summary & Visualization

# %%
# --- EPS comparison across companies ---
if not tier3_core.empty and 'EarningsPerShareDiluted' in tier3_core.columns:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: EPS across companies
    ax1 = axes[0]
    for ticker in features_by_ticker:
        try:
            eps = tier3_core.loc[ticker, 'EarningsPerShareDiluted'].dropna()
            if not eps.empty:
                eps.plot(ax=ax1, label=ticker, lw=2)
        except (KeyError, TypeError):
            pass
    ax1.set_title('Diluted EPS (Split-Adjusted) — All Companies', fontsize=14)
    ax1.legend(loc='upper left', ncol=2)
    ax1.set_xlabel('')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: P/E ratio
    ax2 = axes[1]
    if 'pe_ratio' in tier3_core.columns:
        for ticker in features_by_ticker:
            try:
                pe = tier3_core.loc[ticker, 'pe_ratio'].dropna()
                pe = pe[(pe > 0) & (pe < 200)]  # Filter outliers
                if not pe.empty:
                    pe.plot(ax=ax2, label=ticker, lw=2)
            except (KeyError, TypeError):
                pass
        ax2.set_title('P/E Ratio — All Companies', fontsize=14)
        ax2.legend(loc='upper left', ncol=2)
        ax2.set_xlabel('')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary

# %%
print("\n" + "=" * 70)
print("📋 PIPELINE SUMMARY")
print("=" * 70)

output_files = [
    ('fundamentals_panel.parquet', 'Tier 3: Core features (dense, ML-ready)'),
    ('fundamentals_all_tags.parquet', 'Tier 2: ALL tags (sparse, for feature selection)'),
    ('stock_splits.parquet', 'Stock splits lookup table'),
    ('price_data.parquet', 'Historical prices'),
]

for fname, desc in output_files:
    fpath = data_path / fname
    if fpath.exists():
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  ✅ {fname:<40} {size_mb:>6.1f} MB  — {desc}")
    else:
        print(f"  ❌ {fname:<40} {'N/A':>6}     — {desc}")

# Per-ticker files
print(f"\n  Per-ticker raw data (Tier 1):")
for ticker in sorted(features_by_ticker.keys()):
    fpath = data_path / f'{ticker}_nums.parquet'
    if fpath.exists():
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"    ✅ {fpath.name:<35} {size_mb:>6.1f} MB")

# Panel summary
if not tier3_core.empty:
    print(f"\n  Panel structure: MultiIndex (ticker, date)")
    print(f"  Companies: {tier3_core.index.get_level_values('ticker').nunique()}")
    print(f"  Date range: {tier3_core.index.get_level_values('date').min().strftime('%Y-%m-%d')} → "
          f"{tier3_core.index.get_level_values('date').max().strftime('%Y-%m-%d')}")
    print(f"  Core features: {len(tier3_core.columns)}")
    print(f"  Total rows: {len(tier3_core):,}")

print("\n" + "=" * 70)
print("💡 Usage examples:")
print("=" * 70)
print("""
  import pandas as pd
  
  # Load the ML-ready panel
  panel = pd.read_parquet('data/fundamentals_panel.parquet')
  
  # Get features for sklearn
  X = panel.drop('pe_ratio', axis=1, errors='ignore').values
  
  # Train/test split by date
  cutoff = '2020-01-01'
  train = panel.loc[panel.index.get_level_values('date') < cutoff]
  test  = panel.loc[panel.index.get_level_values('date') >= cutoff]
  
  # Cross-sectional features (compare companies at same date)
  panel.loc[pd.IndexSlice[:, '2020-03-31'], :]
  
  # Time-series features (lagged values per company)
  panel.groupby('ticker')['EarningsPerShareDiluted'].shift(4)
  
  # Rolling TTM metrics
  panel.groupby('ticker')['NetIncomeLoss'].rolling(4).sum()
  
  # Load ALL tags for feature selection experiments
  all_tags = pd.read_parquet('data/fundamentals_all_tags.parquet')
""")
