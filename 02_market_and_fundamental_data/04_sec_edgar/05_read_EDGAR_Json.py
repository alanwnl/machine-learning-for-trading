import pandas as pd
import json

file_path = 'data/tesla_facts.json'

print(f"Reading data from {file_path}...")
try:
    with open(file_path, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {file_path}. Please check the path.")
    exit()

# Extract all us-gaap items
us_gaap_data = data.get("facts", {}).get("us-gaap", {})

# %%
# 1. GENERAL US-GAAP DATA POINT COUNTING
us_gaap_counts = []
for concept, details in us_gaap_data.items():
    units = details.get("units", {})
    if not units:
        us_gaap_counts.append({
            "Concept": concept,
            "Label": details.get("label", ""),
            "Description": details.get("description", ""),
            "Unit Type": "None",
            "Total Data Points": 0
        })
    else:
        for unit_name, unit_list in units.items():
            us_gaap_counts.append({
                "Concept": concept,
                "Label": details.get("label", ""),
                "Description": details.get("description", ""),
                "Unit Type": unit_name,
                "Total Data Points": len(unit_list) 
            })

df_us_gaap = pd.DataFrame(us_gaap_counts)
df_us_gaap = df_us_gaap.sort_values(by='Total Data Points', ascending=False).reset_index(drop=True)

print(f"\nTotal 'us-gaap' Concepts found: {len(us_gaap_data)}")
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', 50)

print("\n--- US-GAAP Unit Data Point Counts (Top 50) ---")
print(df_us_gaap.head(50).to_string(index=False))

# csv_path = "data/tsla_us_gaap_counts.csv"
# df_us_gaap.to_csv(csv_path, index=False)
# print(f"\nSaved the full US-GAAP units overview to {csv_path}")

# %%
# Extract key quarterly metrics
gross_profit_df = pd.DataFrame(data["facts"]["us-gaap"]["GrossProfit"]["units"]["USD"])
sales_revenue_df = pd.DataFrame(data["facts"]["us-gaap"]["Revenues"]["units"]["USD"])
cost_of_revenues_df = pd.DataFrame(data["facts"]["us-gaap"]["CostOfRevenue"]["units"]["USD"])

# %%
# get just values reported in valid frame, and derive Q4 where missing
def derive_missing_q4(df):
    """
    Computes missing Q4 data from SEC EDGAR facts DataFrame by subtracting 
    Q1, Q2, Q3 values from the Full Year (FY) value, then returns ONLY
    the continuous quarterly data (filters out FY rows).
    """
    # 1. Filter for rows that actually contain a frame
    df = df[df['frame'].notna()].copy()
    
    # 2. Extract year and period type (FY or Q1/Q2/Q3/Q4)
    df['year'] = df['frame'].str[2:6]
    df['period'] = df['frame'].str[6:8]
    df.loc[df['frame'].str.len() == 6, 'period'] = 'FY'
    
    new_q4_rows = []
    
    # 3. Iterate over the extracted years
    for year, group in df.groupby('year'):
        periods_present = set(group['period'])
        
        # If FY is present, Q1-Q3 are present, but Q4 is missing
        if 'FY' in periods_present and 'Q4' not in periods_present:
            if {'Q1', 'Q2', 'Q3'}.issubset(periods_present):
                
                fy_val = group[group['period'] == 'FY']['val'].iloc[0]
                q1_val = group[group['period'] == 'Q1']['val'].iloc[0]
                q2_val = group[group['period'] == 'Q2']['val'].iloc[0]
                q3_val = group[group['period'] == 'Q3']['val'].iloc[0]
                
                q4_val = fy_val - (q1_val + q2_val + q3_val)
                
                # Copy the FY row to inherit structure, accn, form, filed dates, etc.
                q4_row = group[group['period'] == 'FY'].iloc[0].copy()
                q4_row['val'] = q4_val
                q4_row['frame'] = f"CY{year}Q4"
                q4_row['fp'] = 'Q4'
                q4_row['start'] = f"{year}-10-01"
                q4_row['end'] = f"{year}-12-31"
                
                new_q4_rows.append(q4_row)
                
    # 4. Clean up the temporary helper columns
    df = df.drop(columns=['year', 'period'])
    
    # 5. Append new Q4 rows 
    if new_q4_rows:
        new_q4_df = pd.DataFrame(new_q4_rows)
        new_q4_df = new_q4_df.drop(columns=['year', 'period'])
        df = pd.concat([df, new_q4_df], ignore_index=True)
        
    # 6. FILTER OUT FY rows so we ONLY plot quarterly data without crazy offset spikes
    df_quarterly = df[df['frame'].str.len() == 8].copy()
    
    return df_quarterly.sort_values(by='frame').reset_index(drop=True)

# Clean all three metrics
gross_profit_clean = derive_missing_q4(gross_profit_df)[['frame', 'end', 'val']].rename(columns={'val': 'gross_profit'})
sales_revenue_clean = derive_missing_q4(sales_revenue_df)[['frame', 'val']].rename(columns={'val': 'sales_revenue'})
cost_of_revenues_clean = derive_missing_q4(cost_of_revenues_df)[['frame', 'val']].rename(columns={'val': 'cost_of_revenues'})

# Merge everything into one dataframe for combined analysis
combined_df = gross_profit_clean.merge(
    sales_revenue_clean, on='frame', how='outer'
).merge(
    cost_of_revenues_clean, on='frame', how='outer'
)

combined_df = combined_df.sort_values(by='frame').reset_index(drop=True)

combined_df['gross_profit_calculated'] = (
    combined_df['sales_revenue'] - combined_df['cost_of_revenues']
)

# Optional: quick sanity check (how close is it to official GrossProfit?)
combined_df['gp_diff'] = (
    combined_df['gross_profit'] - combined_df['gross_profit_calculated']
)
print("✅ Revenues - CostOfRevenue column added!")
print(f"Mean difference vs official GrossProfit: ${combined_df['gp_diff'].mean():,.0f}")
print(combined_df[['frame', 'sales_revenue', 'cost_of_revenues', 
                   'gross_profit', 'gross_profit_calculated', 'gp_diff']].tail())

combined_df
# %%
import plotly.express as px
pd.options.plotting.backend = "plotly" 

# Plot all metrics on the same graph! Plotly knows how to handle multiple columns easily
combined_df.set_index("end")[['gross_profit', 'sales_revenue', 'cost_of_revenues' ,'gross_profit_calculated']].plot(
    title=f"TESLA: Quarterly Financial Metrics",
    labels= {
        "value": "Value ($)",
        "end": "Quarter End",
        "variable": "Metric"
    }
)



# %%
# ==========================================
# 2. SPECIFIC STOCK SPLIT DETECTION
# ==========================================
# print("\n--- Stock Split Detection ---")
# split_concept = "StockholdersEquityNoteStockSplitConversionRatio1"

# if split_concept in us_gaap_data:
#     split_details = us_gaap_data[split_concept]
#     units = split_details.get("units", {})
    
#     unique_splits = set() # Using a set to deduplicate multiple filings reporting the same event
    
#     for unit_name, data_points in units.items():
#         for point in data_points:
#             # Look for an "instant" in time: has an 'end' date, but NO 'start' date
#             if "end" in point and "start" not in point:
#                 date = point["end"]
#                 ratio = point["val"]
#                 unique_splits.add((date, ratio))
    
#     if unique_splits:
#         # Sort chronologically by date
#         sorted_splits = sorted(list(unique_splits), key=lambda x: x[0])
#         for split_date, ratio in sorted_splits:
#             print(f"[*] Detected {int(ratio)}-for-1 Stock Split declared on: {split_date}")
#     else:
#         print("Stock split concept found, but no exact declaration dates were detected.")
# else:
#     print(f"Concept '{split_concept}' not found in the dataset.")