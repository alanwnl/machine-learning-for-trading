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



# %% [markdown]
#  ## 1. Filtering and Data Prep
#  First, we filter the raw dataset for the specific accounting tag we want (`GrossProfit`)
#  and convert the period end dates (`ddate`) to standard datetime objects.
#  We also filter for `dimn == 0` which represents the "Consolidated" company total,
#  ensuring we aren't accidentally pulling segmented business unit data.

# %%
def extract_quarterly_data(api_df: pd.DataFrame, tag_name: str, dimn: int = 0) -> pd.DataFrame:
    """
    Extracts explicitly reported 1-quarter data and derives missing Q4 data 
    for a given accounting tag and dimension.
    """
    # Filter for the tag and prepare dates
    feature_df = api_df[api_df['tag'] == tag_name].copy()
    feature_df['ddate'] = pd.to_datetime(feature_df['ddate'])
    
    # Filter and sort
    plot_df = feature_df[feature_df['dimn'] == dimn].copy()
    plot_df = plot_df.sort_values(['ddate', 'adsh']).drop_duplicates(subset=['ddate', 'qtrs'], keep='first')
    
    # Separate explicitly reported quarters
    clean_df = plot_df[plot_df['qtrs'] == 1].copy()
    
    # Derive missing Q4 data
    years = plot_df['ddate'].dt.year.unique()
    q4_rows = []
    
    for year in years:
        annual_data = plot_df[(plot_df['ddate'].dt.year == year) & (plot_df['qtrs'] == 4)]
        q3_ytd_data = plot_df[(plot_df['ddate'].dt.year == year) & (plot_df['qtrs'] == 3)]
        
        if not annual_data.empty and not q3_ytd_data.empty:
            annual_date = annual_data['ddate'].iloc[0] 
            annual_val = annual_data['value'].iloc[0]
            q3_ytd_val = q3_ytd_data['value'].iloc[0]
                    
            q4_val = annual_val - q3_ytd_val
            
            # Check if explicitly reported Q4 data exists 
            explicit_q4_match = clean_df[clean_df['ddate'] == annual_date]
            if not explicit_q4_match.empty:
                explicit_q4_val = explicit_q4_match['value'].iloc[0]
                if abs(explicit_q4_val - q4_val) < 1.0: 
                    continue
                else:
                    print(f"[{tag_name}] Mismatch for {annual_date.strftime('%Y-%m-%d')}: Explicit Q4 = {explicit_q4_val}, Derived Q4 = {q4_val}. Keeping both for review.")
                    
            q4_rows.append({
                'ddate': annual_date,
                'value': q4_val,
                'qtrs': 1,
                'dimn': dimn,
                'note': 'Derived Q4 (Annual - Q3 YTD)'
            })
            
    if q4_rows:
        q4_df = pd.DataFrame(q4_rows)
        clean_df = pd.concat([clean_df, q4_df], ignore_index=True)
        
    return clean_df.sort_values('ddate').reset_index(drop=True)

# %%
# Extract continuous quarterly data for GrossProfit
clean_df = extract_quarterly_data(df, 'GrossProfit', dimn=0)

print("Cleaned Data Preview (Includes Derived Q4 data):")
print(clean_df[['ddate', 'value', 'qtrs', 'note'] if 'note' in clean_df.columns else ['ddate', 'value', 'qtrs']].to_string())


auto_rev_df = extract_quarterly_data(df, 'SalesRevenueAutomotive', dimn=0)
auto_cost_df = extract_quarterly_data(df, 'CostOfRevenuesAutomotive', dimn=0)

# %% [markdown]
#  ## 4. Visualizing the Continuous Quarterly Series
#  Now that we have a contiguous time series of purely discrete, 3-month quarterly values (`qtrs=1`),
#  we can visualize it cleanly without overlapping or compounding Annual/YTD spikes.

# %%
# Combine them for plotting
clean_df['metric'] = 'Gross Profit'
auto_rev_df['metric'] = 'Automotive Revenue'
auto_cost_df['metric'] = 'Automotive Cost'

# Calculate derived Automotive Gross Profit (Rev - Cost)
auto_gp_df = auto_rev_df.merge(auto_cost_df[['ddate', 'value']], on='ddate', suffixes=('', '_cost'))
auto_gp_df['value'] = auto_gp_df['value'] - auto_gp_df['value_cost']
auto_gp_df['metric'] = 'Auto Gross Profit (Derived)'
auto_gp_df = auto_gp_df.drop(columns=['value_cost'])

plot_all_df = pd.concat([clean_df, auto_rev_df, auto_cost_df, auto_gp_df])

# Plot the final, clean discrete quarterly data for all three metrics
fig = px.line(
    plot_all_df,
    x='ddate',
    y='value',
    color='metric',
    title='Tesla Quarterly Financials: Rev vs Cost vs Gross Profit (Adjusted for Q4)',
    labels={'ddate': 'Period End Date', 'value': 'Reported Value (USD)', 'metric': 'Metric'},
    markers=True,
    template='plotly_white'
)
fig.show()


