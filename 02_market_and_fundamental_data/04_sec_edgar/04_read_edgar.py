# # # %pip install edgartools
# # %pip install --upgrade edgartools
# # %pip install --upgrade pip

# # from edgar import *
# # set_identity("wongalan2002@@yahoo.com.hk")
# # # %%
# # # Get a company's balance sheet
# # balance_sheet = Company("AAPL").get_financials().balance_sheet()
# # balance_sheet

# # # %%
# # # Browse a company's filings
# # company = Company("MSFT")
# # company
# # # %%
# # # Parse insider transactions
# # filings = company.get_filings(form="4")
# # form4 = filings[0].obj()
# # form4

# # from edgar import *
# # set_identity("your.email@example.com")   # required by SEC

# # company = Company("TSLA")

# # # # Get the most recent 10-Q or 10-K
# # # filing = company.latest("10-Q")          # or "10-K"

# # # # Parse the XBRL
# # # xb = filing.xbrl()

# # # # Query by label (this is the magic for post-2021 data)
# # # auto_sales_rev = xb.query().by_label("Automotive sales").to_dataframe()
# # # auto_reg_credits = xb.query().by_label("Automotive regulatory credits").to_dataframe()
# # # auto_leasing_rev = xb.query().by_label("Automotive leasing").to_dataframe()

# # # auto_sales_cost = xb.query().by_label("Automotive sales", exact=False).to_dataframe()  # cost lines also contain "Automotive sales"
# # # # (you can further filter by statement_type or context)

# # # print(auto_sales_rev[['period_end', 'value', 'label']])


# from edgar import Company, set_identity

# set_identity("wongalan2002@yahoo.com.hk")

# tesla = Company("TSLA")
# filing = tesla.latest("10-K")
# xb = filing.xbrl()

# # Use by_statement_type instead of by_statement
# auto_facts_df = (xb.query()
#     .by_statement_type("IncomeStatement") 
#     .by_label("Automotive", exact=False) 
#     .to_dataframe("concept", "label", "value", "period_end")
# )

# auto_facts_df


# import pandas as pd
# from edgar import Company, set_identity

# set_identity("wongalan2002@yahoo.com.hk")

# tesla = Company("TSLA")
# filing = tesla.latest("10-K")
# xb = filing.xbrl()

# income_stmt = xb.statements.income_statement()
# df = income_stmt.to_dataframe(view="detailed")

# auto_facts_df = df[df['label'].str.contains('Automotive', case=False, na=False)]

# auto_facts_df


import pandas as pd
from edgar import Company, set_identity

# Set your SEC identity
set_identity("wongalan2002@yahoo.com.hk")

tesla = Company("TSLA")

# 1. Fetch the entire history of 10-K and 10-Q filings
print("Fetching filing history...")
filings = tesla.get_filings(form=["10-K", "10-Q"])

all_historical_data = []

# 2. Loop through each filing
for filing in filings:
    print(f"Processing {filing.form} filed on {filing.filing_date}...")
    
    try:
        xb = filing.xbrl()
        if not xb: 
            continue # Skip if no XBRL is attached (common in very old SEC filings)
            
        income_stmt = xb.statements.income_statement()
        if not income_stmt:
            continue
            
        df = income_stmt.to_dataframe(view="detailed")
        
        # Filter for Automotive labels
        auto_facts_df = df[df['label'].str.contains('Automotive', case=False, na=False)].copy()
        
        if auto_facts_df.empty:
            continue
            
        # 3. Melt the wide table into a long format for this specific filing
        metadata_cols = ['concept', 'label', 'level', 'abstract']
        # Dynamically find the date columns for this specific filing
        date_columns = [col for col in auto_facts_df.columns if col not in metadata_cols]
        
        # Ensure we only try to keep metadata columns that actually exist in the dataframe
        id_vars = [col for col in metadata_cols if col in auto_facts_df.columns]
        
        long_df = auto_facts_df.melt(
            id_vars=id_vars,
            value_vars=date_columns,
            var_name='ddate',
            value_name='value'
        )
        
        # Drop empty values
        long_df = long_df.dropna(subset=['value']).copy()
        
        # Add tracking metadata so you know where the data came from
        long_df['form'] = filing.form
        long_df['filing_date'] = filing.filing_date
        long_df['accession_no'] = filing.accession_no
        
        all_historical_data.append(long_df)
        
    except Exception as e:
        # Catch errors so one weird filing doesn't crash the whole loop
        print(f"  -> Skipped {filing.accession_no} due to error: {e}")

# 4. Combine all the individual filing dataframes into one massive master table
if all_historical_data:
    master_df = pd.concat(all_historical_data, ignore_index=True)
    
    # 5. Deduplicate! 
    # Because a 10-K reports 3 years of data, the 2023 10-K and 2022 10-K 
    # will both contain the 2022 numbers. We drop the redundant rows here.
    master_df = master_df.drop_duplicates(subset=['concept', 'ddate', 'value'], keep='last')
    
    # Sort chronologically for clean time-series data
    master_df['ddate'] = pd.to_datetime(master_df['ddate'], errors='coerce')
    master_df = master_df.sort_values(by=['concept', 'ddate']).reset_index(drop=True)
    
    print("\nExtraction Complete! Data preview:")
    print(master_df[['concept', 'label', 'ddate', 'value', 'form']].head(10))
    
    # Optional: Save it to your parquet pipeline
    # master_df.to_parquet('data/TSLA_auto_history.parquet')
else:
    print("No historical data could be extracted.")