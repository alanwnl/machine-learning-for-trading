# %% [markdown]
# # How to transform data into factors

# %% [markdown]
# Based on a conceptual understanding of key factor categories, their rationale and popular metrics, a key task is to identify new factors that may better capture the risks embodied by the return drivers laid out previously, or to find new ones. 
# 
# In either case, it will be important to compare the performance of innovative factors to that of known factors to identify incremental signal gains.

# %% [markdown]
# We create the dataset here and store it in our [data](../data) folder to facilitate reuse in later chapters.

# %% [markdown]
# ## Imports & Settings

# %%
# %pip install statsmodels

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
%matplotlib inline

from datetime import datetime
import pandas as pd
import pandas_datareader.data as web

# replaces pyfinance.ols.PandasRollingOLS (no longer maintained)
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

# %%
sns.set_style('whitegrid')
idx = pd.IndexSlice

# %% [markdown]
# ## Get Data

# %% [markdown]
# The `assets.h5` store can be generated using the the notebook [create_datasets](../data/create_datasets.ipynb) in the [data](../data) directory in the root directory of this repo for instruction to download the following dataset.

# %% [markdown]
# We load the Quandl stock price datasets covering the US equity markets 2000-18 using `pd.IndexSlice` to perform a slice operation on the `pd.MultiIndex`, select the adjusted close price and unpivot the column to convert the DataFrame to wide format with tickers in the columns and timestamps in the rows:

# %% [markdown]
# Set data store location:

# %%
DATA_STORE = '../data/assets.h5'

# %%
START = 2000
END = 2018

# %%
with pd.HDFStore(DATA_STORE) as store:
    prices = (store['quandl/wiki/prices']
              .loc[idx[str(START):str(END), :], 'adj_close']
              .unstack('ticker'))
    stocks = store['us_equities/stocks'].loc[:, ['marketcap', 'ipoyear', 'sector']]

# %%
prices.info()
prices

# %%
stocks.info()
stocks

# %% [markdown]
# ### Keep data with stock info

# %% [markdown]
# Remove `stocks` duplicates and align index names for later joining.

# %%
stocks = stocks[~stocks.index.duplicated()]
stocks.index.name = 'ticker'

# %% [markdown]
# Get tickers with both price information and metdata

# %%
shared = prices.columns.intersection(stocks.index)

# %%
stocks = stocks.loc[shared, :]
stocks.info()
stocks

# %%
prices = prices.loc[:, shared]
prices.info()
prices

# %%
assert prices.shape[1] == stocks.shape[0]

# %% [markdown]
# ## Create monthly return series

# %% [markdown]
# To reduce training time and experiment with strategies for longer time horizons, we convert the business-daily data to month-end frequency using the available adjusted close price:

# %%
monthly_prices = prices.resample('M').last()

# %% [markdown]
# To capture time series dynamics that reflect, for example, momentum patterns, we compute historical returns using the method `.pct_change(n_periods)`, that is, returns over various monthly periods as identified by lags.
# 
# We then convert the wide result back to long format with the `.stack()` method, use `.pipe()` to apply the `.clip()` method to the resulting `DataFrame`, and winsorize returns at the [1%, 99%] levels; that is, we cap outliers at these percentiles.
# 
# Finally, we normalize returns using the geometric average. After using `.swaplevel()` to change the order of the `MultiIndex` levels, we obtain compounded monthly returns for six periods ranging from 1 to 12 months:

# %%
monthly_prices.info()
monthly_prices

# %%
outlier_cutoff = 0.01
data = pd.DataFrame()
lags = [1, 2, 3, 6, 9, 12]
for lag in lags:
    data[f'return_{lag}m'] = (monthly_prices
                           .pct_change(lag)
                           .stack()
                           .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                  upper=x.quantile(1-outlier_cutoff)))
                           .add(1)
                           .pow(1/lag)
                           .sub(1)
                           )
data = data.swaplevel().dropna()
data.info()
data

# %% [markdown]
# ## Drop stocks with less than 10 yrs of returns

# %%
min_obs = 120
nobs = data.groupby(level='ticker').size()
keep = nobs[nobs>min_obs].index

data = data.loc[idx[keep,:], :]
data.info()
data

# %%
data.describe()

# %% [markdown]
# This code is visualizing the **Spearman correlation matrix** of the return features (`return_1m`, `return_2m`, `return_3m`, `return_6m`, `return_9m`, `return_12m`) using a **cluster map** from Seaborn.
# 
# ### What does the code do?
# 
# -   `data.corr('spearman')`: Computes the Spearman rank correlation between all columns in [data](vscode-file://vscode-app/Applications/Visual%20Studio%20Code.app/Contents/Resources/app/out/vs/code/electron-sandbox/workbench/workbench.html). Spearman correlation measures monotonic relationships, not just linear ones.
# -   `sns.clustermap(...)`: Plots a heatmap of the correlation matrix, clustering similar rows/columns together.
# -   `annot=True`: Annotates each cell with the correlation value.
# -   `center=0`: Centers the colormap at zero, so positive and negative correlations are visually distinct.
# -   `cmap='Blues'`: Uses a blue color palette for the heatmap.
# 
# ### How to interpret the clustermap?
# 
# -   **Cells**: Each cell shows the correlation between two return features (e.g., `return_1m` vs `return_3m`).
# -   **Color intensity**: Darker blue means higher positive correlation; lighter means lower or negative correlation.
# -   **Annotations**: The number in each cell is the actual Spearman correlation coefficient.
# -   **Clustering**: Rows and columns are reordered so that features with similar correlation patterns are grouped together, making it easier to spot groups of highly correlated features.
# 
# **In summary:**\
# The clustermap helps you quickly see which return features are most strongly related (positively or negatively) and how they cluster together, which is useful for feature selection and understanding redundancy in your data.
# 
# 
# If the **Spearman correlation** between `return_6m` and `return_9m` is **0.77**, while the correlation between `return_6m` and `return_1m` is **0.36**, it means:
# 
# -   **`return_6m` and `return_9m`** move much more closely together (their ranks are more similar across the dataset).
# -   **`return_6m` and `return_1m`** are less closely related (their ranks are less similar).
# 
# **In this context:**\
# Returns over similar time horizons (6 and 9 months) are more strongly related than returns over very different horizons (6 months vs 1 month). This is expected: longer-term returns share more overlapping data and trends, while short-term returns can be more volatile and less related to longer-term trends.
# 
# **Practical implication:**
# 
# -   Features like `return_6m` and `return_9m` may be somewhat redundant (highly correlated).
# -   Features like `return_6m` and `return_1m` capture more independent information.
# 
# This helps you decide which features to keep or combine when building models, to avoid redundancy and multicollinearity.

# %%
# cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(data.corr('spearman'), annot=True, center=0, cmap='Blues');

# %% [markdown]
# We are left with 1,670 tickers.

# %%
data.index.get_level_values('ticker').nunique()

# %% [markdown]
# ## Rolling Factor Betas

# %% [markdown]
# We will introduce the Fama—French data to estimate the exposure of assets to common risk factors using linear regression in [Chapter 9, Time Series Models](../09_time_series_models).

# %% [markdown]
# The five Fama—French factors, namely market risk, size, value, operating profitability, and investment have been shown empirically to explain asset returns and are commonly used to assess the risk/return profile of portfolios. Hence, it is natural to include past factor exposures as financial features in models that aim to predict future returns.

# %% [markdown]
# We can access the historical factor returns using the `pandas-datareader` and estimate historical exposures using the `RollingOLS` rolling linear regression functionality in the `statsmodels` library as follows:

# %% [markdown]
# Use Fama-French research factors to estimate the factor exposures of the stock in the dataset to the 5 factors market risk, size, value, operating profitability and investment.

# %%
factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3', 'famafrench', start='2000')[0].drop('RF', axis=1)
factor_data.index = factor_data.index.to_timestamp()
factor_data = factor_data.resample('M').last().div(100)
factor_data.index.name = 'date'
factor_data.info()
factor_data

# %%
factor_data = factor_data.join(data['return_1m']).sort_index()
factor_data.info()
factor_data

# %%
T = 24
betas = (factor_data.groupby(level='ticker',
                             group_keys=False)
         .apply(lambda x: RollingOLS(endog=x.return_1m,
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(T, x.shape[0]-1))
                .fit(params_only=True)
                .params
                .drop('const', axis=1)))
betas

# %%
betas.describe().join(betas.sum(1).describe().to_frame('total'))

# %%
cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.clustermap(betas.corr(), annot=True, cmap=cmap, center=0);

# %%
data = (data
        .join(betas
              .groupby(level='ticker')
              .shift()))
data.info()
data

# %% [markdown]
# ### Impute mean for missing factor betas

# %%
# data.loc[:, factors] = data.groupby('ticker')[factors].apply(lambda x: x.fillna(x.mean()))
data[factors] = data.groupby('ticker')[factors].transform(lambda x: x.fillna(x.mean()))
data.info()
data

# %% [markdown]
# ## Momentum factors

# %% [markdown]
# We can use these results to compute momentum factors based on the difference between returns over longer periods and the most recent monthly return, as well as for the difference between 3 and 12 month returns as follows:

# %%
for lag in [2,3,6,9,12]:
    data[f'momentum_{lag}'] = data[f'return_{lag}m'].sub(data.return_1m)
data[f'momentum_3_12'] = data[f'return_12m'].sub(data.return_3m)
data

# %% [markdown]
# ## Date Indicators

# %%
dates = data.index.get_level_values('date')
data['year'] = dates.year
data['month'] = dates.month
data

# %% [markdown]
# ## Lagged returns

# %% [markdown]
# To use lagged values as input variables or features associated with the current observations, we use the .shift() method to move historical returns up to the current period:

# %%
for t in range(1, 7):
    data[f'return_1m_t-{t}'] = data.groupby(level='ticker').return_1m.shift(t)
data.info()
data

# %% [markdown]
# ## Target: Holding Period Returns

# %% [markdown]
# Similarly, to compute returns for various holding periods, we use the normalized period returns computed previously and shift them back to align them with the current financial features

# %%
for t in [1,2,3,6,12]:
    data[f'target_{t}m'] = data.groupby(level='ticker')[f'return_{t}m'].shift(-t)

# %%
cols = ['target_1m',
        'target_2m',
        'target_3m', 
        'return_1m',
        'return_2m',
        'return_3m',
        'return_1m_t-1',
        'return_1m_t-2',
        'return_1m_t-3']

data[cols].dropna().sort_index().head(10)

# %%
data.info()
data

# %% [markdown]
# ## Create age proxy

# %% [markdown]
# We use quintiles of IPO year as a proxy for company age.

# %%
data = (data
        .join(pd.qcut(stocks.ipoyear, q=5, labels=list(range(1, 6)))
              .astype(float)
              .fillna(0)
              .astype(int)
              .to_frame('age')))
data.age = data.age.fillna(-1)

# %% [markdown]
# ## Create dynamic size proxy

# %% [markdown]
# We use the marketcap information from the NASDAQ ticker info to create a size proxy.

# %%
stocks.info()
stocks

# %% [markdown]
# Market cap information is tied to currrent prices. We create an adjustment factor to have the values reflect lower historical prices for each individual stock:

# %%
size_factor = (monthly_prices
               .loc[data.index.get_level_values('date').unique(),
                    data.index.get_level_values('ticker').unique()]
               .sort_index(ascending=False)
               .pct_change()
               .fillna(0)
               .add(1)
               .cumprod())
size_factor.info()

# %%
msize = (size_factor
         .mul(stocks
              .loc[size_factor.columns, 'marketcap'])).dropna(axis=1, how='all')

# %% [markdown]
# ### Create Size indicator as deciles per period

# %% [markdown]
# Compute size deciles per month:

# %%
data['msize'] = (msize
                 .apply(lambda x: pd.qcut(x, q=10, labels=list(range(1, 11)))
                        .astype(int), axis=1)
                 .stack()
                 .swaplevel())
data.msize = data.msize.fillna(-1)

# %% [markdown]
# ## Combine data

# %%
data = data.join(stocks[['sector']])
data.sector = data.sector.fillna('Unknown')

# %%
data.info()

# %% [markdown]
# ## Store data

# %% [markdown]
# We will use the data again in several later chapters, starting in [Chapter 7 on Linear Models](../07_linear_models).

# %%
with pd.HDFStore(DATA_STORE) as store:
    store.put('engineered_features', data.sort_index().loc[idx[:, :datetime(2018, 3, 1)], :])
    print(store.info())

# %% [markdown]
# ## Create Dummy variables

# %% [markdown]
# For most models, we need to encode categorical variables as 'dummies' (one-hot encoding):

# %%
dummy_data = pd.get_dummies(data,
                            columns=['year','month', 'msize', 'age',  'sector'],
                            prefix=['year','month', 'msize', 'age', ''],
                            prefix_sep=['_', '_', '_', '_', ''])
dummy_data = dummy_data.rename(columns={c:c.replace('.0', '') for c in dummy_data.columns})
dummy_data.info()


