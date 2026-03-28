# %% [markdown]
# # Process and Analyze SEC EDGAR XRBL Data

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
# %matplotlib inline

from pathlib import Path
import json
import pandas_datareader.data as web
import pandas as pd
from pprint import pprint

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# %%
sns.set_style('whitegrid')

# %%
data_path = Path('data')
if not data_path.exists():
    data_path.mkdir()

# %% [markdown]
# ## Metadata json

# %%
try:
    file = data_path / '2018_3' / 'source' / '2018q3_notes-metadata.json'
    with file.open() as f:
        data = json.load(f)
    pprint(data)
except FileNotFoundError:
    print(f"Metadata file {file} not found. Ensure the download script has run.")

# %% [markdown]
# ## Data Organization

# %% [markdown]
# For each quarter, the FSN data is organized into eight file sets that contain information about submissions, numbers, taxonomy tags, presentation, and more. Each dataset consists of rows and fields and is provided as a tab-delimited text file:

# %% [markdown]
# | File | Dataset      | Description                                                 |
# |------|--------------|-------------------------------------------------------------|
# | SUB  | Submission   | Identifies each XBRL submission by company, form, date, etc |
# | TAG  | Tag          | Defines and explains each taxonomy tag                      |
# | DIM  | Dimension    | Adds detail to numeric and plain text data                  |
# | NUM  | Numeric      | One row for each distinct data point in filing              |
# | TXT  | Plain Text   | Contains all non-numeric XBRL fields                        |
# | REN  | Rendering    | Information for rendering on SEC website                    |
# | PRE  | Presentation | Detail on tag and number presentation in primary statements |
# | CAL  | Calculation  | Shows arithmetic relationships among tags                   |

# %% [markdown]
# ## Submission Data

# %% [markdown]
# The latest submission file contains around 6,500 entries.

# %%
try:
    sub = pd.read_parquet(data_path / '2018_3' / 'parquet' / 'sub.parquet')
    sub.info()
except FileNotFoundError:
    print("sub.parquet not found. Skipping info.")
    sub = pd.DataFrame() # dummy dataframe to prevent downstream crash if missing

# %% [markdown]
# ### Get AAPL submission

# %% [markdown]
# The submission dataset contains the unique identifiers required to retrieve the filings: the Central Index Key (CIK) and the Accession Number (adsh). The following shows some of the information about Apple's 2018Q1 10-Q filing:

# %%
if not sub.empty:
    name = 'APPLE INC'
    apple = sub[sub.name == name].T.dropna().squeeze()
    key_cols = ['name', 'adsh', 'cik', 'name', 'sic', 'countryba', 'stprba',
                'cityba', 'zipba', 'bas1', 'form', 'period', 'fy', 'fp', 'filed']
    print(apple.loc[key_cols])

# %% [markdown]
# ## Build AAPL fundamentals dataset

# %% [markdown]
# Using the central index key, we can identify all historical quarterly filings available for Apple, and combine this information to obtain 26 Forms 10-Q and nine annual Forms 10-K.

# %% [markdown]
# ### Get filings

# %%
aapl_subs = pd.DataFrame()
if not sub.empty:
    for s_file in data_path.glob('**/sub.parquet'):
        s_data = pd.read_parquet(s_file)
        aapl_sub = s_data[(s_data.cik.astype(int) == apple.cik) & (s_data.form.isin(['10-Q', '10-K']))]
        aapl_subs = pd.concat([aapl_subs, aapl_sub])

# %% [markdown]
# We find 15 quarterly 10-Q and 4 annual 10-K reports:

# %%
if not aapl_subs.empty:
    print(aapl_subs.form.value_counts())

# %% [markdown]
# ### Get numerical filing data

# %% [markdown]
# With the Accession Number for each filing, we can now rely on the taxonomies to select the appropriate XBRL tags (listed in the TAG file) from the NUM and TXT files to obtain the numerical or textual/footnote data points of interest.

# %% [markdown]
# First, let's extract all numerical data available from the 19 Apple filings:

# %%
aapl_nums = pd.DataFrame()
if not aapl_subs.empty:
    for num in data_path.glob('**/num.parquet'):
        num_data = pd.read_parquet(num).drop('dimh', axis=1, errors='ignore')
        aapl_num = num_data[num_data.adsh.isin(aapl_subs.adsh)]
        print(len(aapl_num))
        aapl_nums = pd.concat([aapl_nums, aapl_num])
        
if not aapl_nums.empty:
    aapl_nums.ddate = pd.to_datetime(aapl_nums.ddate, format='%Y%m%d')   
    aapl_nums.to_parquet(data_path / 'aapl_nums.parquet')

# %% [markdown]
# In total, the nine years of filing history provide us with over 18,000 numerical values for AAPL.

# %%
if not aapl_nums.empty:
    aapl_nums.info()

# %% [markdown]
# ## Create P/E Ratio from EPS and stock price data

# %% [markdown]
# We can select a useful field, such as Earnings per Diluted Share (EPS), that we can combine with market data to calculate the popular Price/Earnings (P/E) valuation ratio.

# %%
stock_split = 7
split_date = pd.to_datetime('20140604')
print(split_date)

# %% [markdown]
# We do need to take into account, however, that Apple split its stock 7:1 on June 4, 2014, and Adjusted Earnings per Share before the split to make earnings comparable, as illustrated in the following code block:

# %%
if not aapl_nums.empty:
    # Filter by tag; keep only values measuring 1 quarter
    eps = aapl_nums[(aapl_nums.tag == 'EarningsPerShareDiluted')
                    & (aapl_nums.qtrs == 1)].drop('tag', axis=1)

    # Keep only most recent data point from each filing
    eps = eps.groupby('adsh').apply(lambda x: x.nlargest(n=1, columns=['ddate']))

    # Adjust earnings prior to stock split downward
    eps.loc[eps.ddate < split_date,'value'] = eps.loc[eps.ddate < split_date, 'value'].div(7)
    eps = eps[['ddate', 'value']].set_index('ddate').squeeze().sort_index()
    eps = eps.rolling(4,min_periods=4).sum().dropna()

# %%
if 'eps' in locals() and not eps.empty:
    eps.plot(lw=2, figsize=(14, 6), title='Diluted Earnings per Share')
    plt.xlabel('')
    plt.savefig('diluted eps', dpi=300)

# %%
symbol = 'AAPL'

if 'eps' in locals() and not eps.empty:
    import yfinance as yf
    aapl_stock = yf.download(symbol, start=eps.index.min())
    aapl_stock.index = pd.to_datetime(aapl_stock.index).tz_localize(None)
    aapl_stock = (aapl_stock
                  .resample('D')
                  .last()
                  .loc['2014':eps.index.max()])
    
    if isinstance(aapl_stock.columns, pd.MultiIndex):
        aapl_stock.columns = aapl_stock.columns.get_level_values(0)
        
    if 'Adj Close' in aapl_stock.columns:
        aapl_stock = aapl_stock.rename(columns={'Adj Close': 'AdjClose'})
    elif 'Close' in aapl_stock.columns and 'AdjClose' not in aapl_stock.columns:
        aapl_stock = aapl_stock.rename(columns={'Close': 'AdjClose'})

    aapl_stock.info()

# %%
if 'aapl_stock' in locals() and not aapl_stock.empty and 'eps' in locals() and not eps.empty:
    pe = aapl_stock.AdjClose.to_frame('price').join(eps.to_frame('eps'))
    pe = pe.fillna(method='ffill').dropna()
    pe['P/E Ratio'] = pe.price.div(pe.eps)
    pe['P/E Ratio'].plot(lw=2, figsize=(14, 6), title='TTM P/E Ratio')

# %%
if 'pe' in locals() and not pe.empty:
    pe.info()

# %%
if 'pe' in locals() and not pe.empty:
    axes = pe.plot(subplots=True, figsize=(16,8), legend=False, lw=2)
    axes[0].set_title('Adj. Close Price')
    axes[1].set_title('Diluted Earnings per Share')
    axes[2].set_title('Trailing P/E Ratio')
    plt.tight_layout()

# %% [markdown]
# ## Explore Additional Fields

# %% [markdown]
# The field `tag` references values defined in the taxonomy:

# %%
if not aapl_nums.empty:
    print(aapl_nums.tag.value_counts())

# %% [markdown]
# We can select values of interest and track their value or use them as inputs to compute fundamental metrics like the Dividend/Share ratio.

# %% [markdown]
# ### Dividends per Share

# %%
fields = ['EarningsPerShareDiluted',
          'PaymentsOfDividendsCommonStock',
          'WeightedAverageNumberOfDilutedSharesOutstanding',
          'OperatingIncomeLoss',
          'NetIncomeLoss',
          'GrossProfit']

# %%
if not aapl_nums.empty:
    dividends = (aapl_nums
                 .loc[aapl_nums.tag == 'PaymentsOfDividendsCommonStock', ['ddate', 'value']]
                 .groupby('ddate')
                 .mean())
    shares = (aapl_nums
              .loc[aapl_nums.tag == 'WeightedAverageNumberOfDilutedSharesOutstanding', ['ddate', 'value']]
              .drop_duplicates()
              .groupby('ddate')
              .mean())
    df = dividends.div(shares).dropna()
    if not df.empty:
        ax = df.plot.bar(figsize=(14, 5), title='Dividends per Share', legend=False)
        ax.xaxis.set_major_formatter(mticker.FixedFormatter(df.index.strftime('%Y-%m')))

# %% [markdown]
# ## Bonus: Textual Information

# %%
try:
    txt = pd.read_parquet(data_path / '2016_2' / 'parquet' /  'txt.parquet')
except FileNotFoundError:
    print("txt.parquet for 2016_2 not found. Skipping textual information.")
    txt = pd.DataFrame()

# %% [markdown]
# AAPL's adsh is not avaialble in the txt file but you can obtain notes from the financial statements here:

# %%
if not txt.empty:
    print(txt.head())
