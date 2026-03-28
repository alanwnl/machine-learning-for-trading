# %% [markdown]
#  # SEC EDGAR Data Processing: Deriving Clean Quarterly Data
#  This notebook demonstrates how to load raw SEC EDGAR XBRL data from a Parquet file
#  and process it to extract clean, discrete quarterly numbers.
# 
#  A common challenge with SEC data is that Q4 numbers are rarely reported explicitly
#  as a discrete 3-month value (`qtrs=1`) in the 10-K filings. Instead, the 10-K primarily
#  provides the annual total (`qtrs=4`).

# %%
import pandas as pd
import plotly.express as px

# Define the path to your parquet file
parquet_path = 'data/TSLA_nums.parquet'

# Read the Parquet file into a Pandas DataFrame
df = pd.read_parquet(parquet_path)

# %%
# tags = print(df['tag'].unique())
# tags

tag_counts = df['tag'].value_counts()

# print(tag_counts)
print(tag_counts.to_string())
# tag_counts.to_csv('tesla_sec_tags.csv')

# %%
# 1. Get the Automotive Revenue
auto_rev = df[(df['tag'] == 'SalesRevenueAutomotive') & (df['dimn'] == 0)].copy()
auto_rev = auto_rev.sort_values(['ddate', 'adsh']).drop_duplicates(subset=['ddate', 'qtrs'], keep='first')
print(auto_rev.shape)
auto_rev

# %%
# 2. Get the Automotive Cost
auto_cost = df[(df['tag'] == 'CostOfRevenuesAutomotive') & (df['dimn'] == 0)].copy()
auto_cost = auto_cost.sort_values(['ddate', 'adsh']).drop_duplicates(subset=['ddate', 'qtrs'], keep='first')
print(auto_cost.shape)
auto_cost

# %% [markdown]
#  ## 1. Filtering and Data Prep
#  First, we filter the raw dataset for the specific accounting tag we want (`GrossProfit`)
#  and convert the period end dates (`ddate`) to standard datetime objects.
#  We also filter for `dimn == 0` which represents the "Consolidated" company total,
#  ensuring we aren't accidentally pulling segmented business unit data.

# %%
# Filter for 'GrossProfit' and prepare dates
gross_profit_df = df[df['tag'] == 'GrossProfit'].copy()
gross_profit_df['ddate'] = pd.to_datetime(gross_profit_df['ddate'])

# Filter for Consolidated Totals Only (ALL Years)
plot_df = gross_profit_df[gross_profit_df['dimn'] == 0].copy()

# Sort and drop duplicates to ensure we get the first/latest reported numbers if there are restatements
plot_df = plot_df.sort_values(['ddate', 'adsh']).drop_duplicates(subset=['ddate', 'qtrs'], keep='first')

# %% [markdown]
#  ## 2. Isolating Q1, Q2, and Q3
#  Quarters 1, 2, and 3 are typically reported in 10-Q filings with a `qtrs` value of 1
#  (meaning it represents exactly 1 quarter / 3 months of duration). We can extract these directly.

# %%
# Separate Q1, Q2, Q3 (where qtrs == 1 explicitly exists)
clean_df = plot_df[plot_df['qtrs'] == 1].copy()

# %% [markdown]
#  ## 3. Deriving Q4 Data
#  **How to get precise Q4 Data:**
#  Companies do not typically file a standard 10-Q for the 4th quarter. Instead, they file an annual 10-K.
#  Because of this, the `qtrs=1` tag for the period ending on Q4's date (often Dec 31) is usually missing.
# 
#  To systematically calculate the discrete Q4 value, we use the formula:
#  **`Q4 Value = Annual Year-to-Date Value (qtrs=4) - Q3 Year-to-Date Value (qtrs=3)`**
# 
#  The loop below matches the `qtrs=3` row with the `qtrs=4` row for the exact same fiscal year
#  and calculates the discrete Q4 difference.

# %%
# Find all unique years in our dataset
years = plot_df['ddate'].dt.year.unique()
q4_rows = []

for year in years:
    # Find Annual data (qtrs=4) and Q3 YTD data (qtrs=3) for the current year
    annual_data = plot_df[(plot_df['ddate'].dt.year == year) & (plot_df['qtrs'] == 4)]
    q3_ytd_data = plot_df[(plot_df['ddate'].dt.year == year) & (plot_df['qtrs'] == 3)]
    
    # If both exist for this given year, we can systematically calculate Q4
    if not annual_data.empty and not q3_ytd_data.empty:
        annual_val = annual_data['value'].iloc[0]
        q3_ytd_val = q3_ytd_data['value'].iloc[0]
        
        # The date for Q4 is the exact same as the annual report end date
        annual_date = annual_data['ddate'].iloc[0] 
        
        q4_val = annual_val - q3_ytd_val
        
        # Store the calculated Q4 row as a derived qtrs=1 record
        q4_rows.append({
            'ddate': annual_date,
            'value': q4_val,
            'qtrs': 1,
            'dimn': 0,
            'note': 'Derived Q4 (Annual - Q3 YTD)' # Helpful metadata tag for auditing
        })

# Append all the newly calculated Q4 rows back to our clean dataframe
if q4_rows:
    q4_df = pd.DataFrame(q4_rows)
    clean_df = pd.concat([clean_df, q4_df], ignore_index=True)

# Ensure the final dataframe is sorted chronologically
clean_df = clean_df.sort_values('ddate').reset_index(drop=True)

print("Cleaned Data Preview (Includes Derived Q4 data):")
print(clean_df[['ddate', 'value', 'qtrs', 'note'] if 'note' in clean_df.columns else ['ddate', 'value', 'qtrs']].tail(8))

# %% [markdown]
#  ## 4. Visualizing the Continuous Quarterly Series
#  Now that we have a contiguous time series of purely discrete, 3-month quarterly values (`qtrs=1`),
#  we can visualize it cleanly without overlapping or compounding Annual/YTD spikes.

# %%
# Plot the final, clean discrete quarterly data for ALL time
fig = px.line(
    clean_df,
    x='ddate',
    y='value',
    title='Tesla Consolidated Quarterly Gross Profit (All Time) – Adjusted for Q4',
    labels={'ddate': 'Period End Date', 'value': 'Quarterly Gross Profit (USD)'},
    markers=True,
    template='plotly_white'
)
fig.show()


