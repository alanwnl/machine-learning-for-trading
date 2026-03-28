# from pathlib import Path
# import pandas as pd

# data_path = Path('data')

# # Load the file into a DataFrame
# aapl_nums = pd.read_parquet(data_path / 'aapl_nums.parquet')

# # Quick check that it worked
# print("✅ Loaded successfully!")
# print(f"Shape: {aapl_nums.shape} (rows × columns)")
# aapl_nums.info()          # shows column names + data types
# aapl_nums.head()          # shows first 5 rows

# %%

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

aapl_nums = pd.read_parquet(Path('data') / 'aapl_nums.parquet')

# Convert date for easy filtering
aapl_nums['ddate'] = pd.to_datetime(aapl_nums['ddate'], format='%Y%m%d')

# %%
# === SHOW ALL TAGS ===
unique_tags = sorted(aapl_nums['tag'].unique())   # sorted alphabetically

print(f"✅ Total unique tags: {len(unique_tags):,}\n")
print("All tags (sorted):")
for tag in unique_tags:
    print(tag)
# %%
# Example 1: Get all Earnings Per Share values
eps = aapl_nums[aapl_nums.tag == 'EarningsPerShareDiluted']

# %%
# Example 2: See the 20 most common tags
# === FORCE JUPYTER TO SHOW EVERYTHING ===
pd.set_option('display.max_rows', 200)        # 200 is enough for top 100
pd.set_option('display.max_colwidth', None)   # just in case

# Now show the top 100 most common tags
print("=== Top 100 Most Common Tags in Apple's Filings ===")
print(aapl_nums.tag.value_counts().head(100))


# %%
# Example 3: Create a time series dataframe for CashAndCashEquivalentsAtCarryingValue
cash_tag = 'CashAndCashEquivalentsAtCarryingValue'

# 1. Filter by tag
cash = aapl_nums[aapl_nums.tag == cash_tag].copy()

# 2. Extract value by filing date
# Keep only the most recent data point from each filing (adsh)
cash_ts = cash.groupby('adsh').apply(lambda x: x.nlargest(n=1, columns=['ddate']))

# 3. Set the filing date as index, keeping only 'value', and format as DataFrame
cash_ts = cash_ts[['ddate', 'value']].set_index('ddate').squeeze().sort_index()
cash_df = cash_ts.to_frame('CashAndCashEquivalents')

# Display the resulting time series DataFrame
print(f"\n=== {cash_tag} Time Series ===")
print(cash_df.head())
print(f"Total periods: {len(cash_df)}")

# %%
# 4. Plot the time series
ax = cash_df.plot(lw=2, figsize=(14, 6), title='Apple Inc. Cash and Cash Equivalents (Carrying Value)', legend=False)
ax.set_ylabel('Amount (USD)')
ax.set_xlabel('Filing Date')
plt.tight_layout()
# plt.savefig('apple_cash_equivalents.png', dpi=300)
print("\n✅ Plot saved as 'apple_cash_equivalents.png'")