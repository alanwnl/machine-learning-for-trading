# %% [markdown]
#  # Working with filing data from the SEC's EDGAR service

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
%matplotlib inline

from pathlib import Path
from datetime import date
import json
from io import BytesIO
from zipfile import ZipFile, BadZipFile
from tqdm import tqdm
import requests

import yfinance as yf
import pandas as pd

from pprint import pprint

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# %%
sns.set_style('whitegrid')

# %%
# --- Parameters ---
DATA_DIR = 'data'
START_DATE = '2014'
END_DATE = '2025-12-31'
TARGET_COMPANY = 'APPLE INC'
TARGET_TICKER = 'AAPL' # for yfinance
STOCK_SPLIT = 7
SPLIT_DATE = pd.to_datetime('20140604')
USER_AGENT = 'MachineLearningForTrading_Student admin@ml4t.com'
# ------------------

# %%
# store data in this directory since we won't use it in other chapters
data_path = Path(DATA_DIR) # perhaps set to external harddrive to accomodate large amount of data
if not data_path.exists():
    data_path.mkdir(parents=True)

# %% [markdown]
#  ## Download FS & Notes Data

# %% [markdown]
#  The following code downloads and extracts all historical filings contained in the [Financial Statement and Notes](https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html) (FSN) datasets from Q1/2014 through Q3/2020.
# 
#  > The SEC has moved to a monthly cadence after Q3/2020; feel free to extend the code by creating the correpsonding file names (see linked website) and download those as well.

# %% [markdown]
#  **Downloads over 40GB of data!**

# %%
SEC_URL = 'https://www.sec.gov/'
FSN_PATH = 'files/dera/data/financial-statement-notes-data-sets/'

# %%
filing_periods = [(d.year, d.quarter) for d in pd.date_range(START_DATE, END_DATE, freq='QE')]
filing_periods

# %%
for yr, qtr in tqdm(filing_periods):
    # set (and create) directory
    path = data_path / f'{yr}_{qtr}' / 'source'
    if not path.exists():
        path.mkdir(parents=True)
    
    # define url and get file
    filing = f'{yr}q{qtr}_notes.zip'
    url = SEC_URL + FSN_PATH + filing
    headers = {'User-Agent': USER_AGENT}
    
    if len(list(path.glob('*.*'))) > 0:
        print(f'Already downloaded and extracted {yr} Q{qtr}')
        continue
        
    response = requests.get(url, headers=headers).content
    
    # decompress and save
    try:
        with ZipFile(BytesIO(response)) as zip_file:
            for file in zip_file.namelist():
                local_file = path / file
                if local_file.exists():
                    continue
                with local_file.open('wb') as output:
                    for line in zip_file.open(file).readlines():
                        output.write(line)
    except BadZipFile:
        print(f'\nBad zip file: {yr} {qtr}\n')
        continue

# %% [markdown]
#  ## Save to parquet

# %% [markdown]
#  The data is fairly large and to enable faster access than the original text files permit, it is better to convert the text files to binary, columnar parquet format (see Section 'Efficient data storage with pandas' in chapter 2 for a performance comparison of various data-storage options compatible with pandas DataFrames):

# %% [markdown]
#  > Some fo the `txt.tsv` source files contain a small number of faulty lines; the code below drops those lines but indicates the line numbers where you can find the errors if you would like to investigate further.

# %%
for f in tqdm(sorted(list(data_path.glob('**/*.tsv')))):
    # set (and create) directory
    parquet_path = f.parent.parent / 'parquet'
    if not parquet_path.exists():
        parquet_path.mkdir(parents=True)    

    # write content to .parquet
    file_name = f.stem  + '.parquet'
    if not (parquet_path / file_name).exists():
        try:
            df = pd.read_csv(f, sep='\t', encoding='latin1', low_memory=False, on_bad_lines='skip')
            df.to_parquet(parquet_path / file_name)
        except Exception as e:
            print(e, ' | ', f)
        # optional: uncomment to delete original .tsv
#         else:
            # f.unlink

# %% [markdown]
#  ## Metadata json

# %%
sample_yr, sample_qtr = filing_periods[0]
sample_dir = f'{sample_yr}_{sample_qtr}'
file = data_path / sample_dir / 'source' / f'{sample_yr}q{sample_qtr}_notes-metadata.json'
with file.open() as f:
    data = json.load(f)

pprint(data)

# %% [markdown]
#  ## Data Organization

# %% [markdown]
#  For each quarter, the FSN data is organized into eight file sets that contain information about submissions, numbers, taxonomy tags, presentation, and more. Each dataset consists of rows and fields and is provided as a tab-delimited text file:

# %% [markdown]
#  | File | Dataset      | Description                                                 |
#  |------|--------------|-------------------------------------------------------------|
#  | SUB  | Submission   | Identifies each XBRL submission by company, form, date, etc |
#  | TAG  | Tag          | Defines and explains each taxonomy tag                      |
#  | DIM  | Dimension    | Adds detail to numeric and plain text data                  |
#  | NUM  | Numeric      | One row for each distinct data point in filing              |
#  | TXT  | Plain Text   | Contains all non-numeric XBRL fields                        |
#  | REN  | Rendering    | Information for rendering on SEC website                    |
#  | PRE  | Presentation | Detail on tag and number presentation in primary statements |
#  | CAL  | Calculation  | Shows arithmetic relationships among tags                   |

# %% [markdown]
#  ## Submission Data

# %% [markdown]
#  The latest submission file contains around 6,500 entries.

# %%
sub = pd.read_parquet(data_path / sample_dir / 'parquet' / 'sub.parquet')
sub.info()

# %% [markdown]
#  ### Get AAPL submission

# %% [markdown]
#  The submission dataset contains the unique identifiers required to retrieve the filings: the Central Index Key (CIK) and the Accession Number (adsh). The following shows some of the information about Apple's 2018Q1 10-Q filing:

# %%
apple = sub[sub.name == TARGET_COMPANY].T.dropna().squeeze()
key_cols = ['name', 'adsh', 'cik', 'name', 'sic', 'countryba', 'stprba',
            'cityba', 'zipba', 'bas1', 'form', 'period', 'fy', 'fp', 'filed']
apple.loc[key_cols]

# %% [markdown]
#  ## Build AAPL fundamentals dataset

# %% [markdown]
#  Using the central index key, we can identify all historical quarterly filings available for Apple, and combine this information to obtain 26 Forms 10-Q and nine annual Forms 10-K.

# %% [markdown]
#  ### Get filings

# %%
aapl_subs = pd.DataFrame()
for sub in data_path.glob('**/sub.parquet'):
    sub = pd.read_parquet(sub)
    aapl_sub = sub[(sub.cik.astype(int) == apple.cik) & (sub.form.isin(['10-Q', '10-K']))]
    aapl_subs = pd.concat([aapl_subs, aapl_sub])

# %% [markdown]
#  We find 15 quarterly 10-Q and 4 annual 10-K reports:

# %%
aapl_subs.form.value_counts()

# %% [markdown]
#  ### Get numerical filing data

# %% [markdown]
#  With the Accession Number for each filing, we can now rely on the taxonomies to select the appropriate XBRL tags (listed in the TAG file) from the NUM and TXT files to obtain the numerical or textual/footnote data points of interest.

# %% [markdown]
#  First, let's extract all numerical data available from the 19 Apple filings:

# %%
aapl_nums = pd.DataFrame()
for num in data_path.glob('**/num.parquet'):
    num = pd.read_parquet(num).drop('dimh', axis=1)
    aapl_num = num[num.adsh.isin(aapl_subs.adsh)]
    print(len(aapl_num))
    aapl_nums = pd.concat([aapl_nums, aapl_num])
aapl_nums.ddate = pd.to_datetime(aapl_nums.ddate, format='%Y%m%d')   
aapl_nums.to_parquet(data_path / 'aapl_nums.parquet')

# %% [markdown]
#  In total, the nine years of filing history provide us with over 18,000 numerical values for AAPL.

# %%
aapl_nums.info()

# %% [markdown]
#  ## Create P/E Ratio from EPS and stock price data

# %% [markdown]
#  We can select a useful field, such as Earnings per Diluted Share (EPS), that we can combine with market data to calculate the popular Price/Earnings (P/E) valuation ratio.

# %%
SPLIT_DATE

# %% [markdown]
#  We do need to take into account, however, that Apple split its stock 7:1 on June 4, 2014, and Adjusted Earnings per Share before the split to make earnings comparable, as illustrated in the following code block:

# %%
# Filter by tag; keep only values measuring 1 quarter
eps = aapl_nums[(aapl_nums.tag == 'EarningsPerShareDiluted')
                & (aapl_nums.qtrs == 1)].drop('tag', axis=1)

# Keep only most recent data point from each filing
eps = eps.groupby('adsh').apply(lambda x: x.nlargest(n=1, columns=['ddate']))

# Adjust earnings prior to stock split downward
eps.loc[eps.ddate < SPLIT_DATE,'value'] = eps.loc[eps.ddate < SPLIT_DATE, 'value'].div(STOCK_SPLIT)
eps = eps[['ddate', 'value']].set_index('ddate').squeeze().sort_index()
eps = eps.rolling(4,min_periods=4).sum().dropna()

# %%
eps.plot(lw=2, figsize=(14, 6), title='Diluted Earnings per Share')
plt.xlabel('')
plt.savefig('diluted eps', dpi=300);

# %%
aapl_stock = yf.Ticker(TARGET_TICKER).history(start=eps.index.min())
aapl_stock.index = aapl_stock.index.tz_localize(None)
aapl_stock = (aapl_stock
              .rename(columns={'Close': 'AdjClose'})
              .resample('D')
              .last()
             .loc[START_DATE:eps.index.max()])
aapl_stock.info()

# %%
pe = aapl_stock.AdjClose.to_frame('price').join(eps.to_frame('eps'))
pe = pe.ffill().dropna()
pe['P/E Ratio'] = pe.price.div(pe.eps)
pe['P/E Ratio'].plot(lw=2, figsize=(14, 6), title='TTM P/E Ratio');

# %%
pe.info()

# %%
axes = pe.plot(subplots=True, figsize=(16,8), legend=False, lw=2)
axes[0].set_title('Adj. Close Price')
axes[1].set_title('Diluted Earnings per Share')
axes[2].set_title('Trailing P/E Ratio')
plt.tight_layout();

# %% [markdown]
#  ## Explore Additional Fields

# %% [markdown]
#  The field `tag` references values defined in the taxonomy:

# %%
aapl_nums.tag.value_counts()

# %% [markdown]
#  We can select values of interest and track their value or use them as inputs to compute fundamental metrics like the Dividend/Share ratio.

# %% [markdown]
#  ### Dividends per Share

# %%
fields = ['EarningsPerShareDiluted',
          'PaymentsOfDividendsCommonStock',
          'WeightedAverageNumberOfDilutedSharesOutstanding',
          'OperatingIncomeLoss',
          'NetIncomeLoss',
          'GrossProfit']

# %%
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
df.index = df.index.strftime('%Y-%m')
ax = df.plot.bar(figsize=(14, 5), title='Dividends per Share', legend=False)

# %% [markdown]
#  ## Bonus: Textual Information

# %%
txt = pd.read_parquet(data_path / sample_dir / 'parquet' /  'txt.parquet')

# %% [markdown]
#  AAPL's adsh is not avaialble in the txt file but you can obtain notes from the financial statements here:

# %%
txt.head()


