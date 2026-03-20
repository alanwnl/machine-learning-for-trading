import os
import nasdaqdatalink

# Configuration
# If you make more than 50 calls a day, you will need to create a free account and set your API key.
# API key can be set here directly, or by setting NASDAQ_DATA_LINK_API_KEY environment variable.
nasdaq_api_key = 'JkR_HtG6XFodqy27f9V9'
if nasdaq_api_key:
    nasdaqdatalink.ApiConfig.api_key = nasdaq_api_key

# Alternatively, read from a local key file explicitly:
# nasdaqdatalink.read_key(filepath="~/.nasdaq/data_link_apikey")

def main():
    print("=== Time-Series Format Examples ===")
    
    # 1. Make a Basic Data Call
    print("\n1. Basic Data Call: US Department of Energy WTI Crude Oil price (EIA/PET_RWTC_D)")
    mydata = nasdaqdatalink.get("EIA/PET_RWTC_D")
    print(mydata.head())

    # 2. Change Formats
    print("\n2. Getting the same data as a NumPy array (returns='numpy')")
    mydata_numpy = nasdaqdatalink.get("EIA/PET_RWTC_D", returns="numpy")
    print(mydata_numpy[:5])

    # 3. Slice and Dice the Data
    # Set start and end dates
    print("\n3. Start and end dates: US GDP from FRED (FRED/GDP, 2001-2005)")
    gdp_data = nasdaqdatalink.get("FRED/GDP", start_date="2001-12-31", end_date="2005-12-31")
    print(gdp_data.head())

    # Request specific columns
    # Note: "WIKI" dataset has been deprecated but included as example
    # mydata_cols = nasdaqdatalink.get(["NSE/OIL.1", "WIKI/AAPL.4"])
    
    # Request the last n rows
    print("\n4. Request last 5 rows: WTI Crude Oil price")
    mydata_last_n = nasdaqdatalink.get("EIA/PET_RWTC_D", rows=5)
    print(mydata_last_n)

    # 4. Preprocess the Data
    # Change the sampling frequency
    print("\n5. Change sampling frequency (collapse='monthly')")
    monthly_data = nasdaqdatalink.get("EIA/PET_RWTC_D", collapse="monthly")
    print(monthly_data.head())

    # Perform elementary calculations
    print("\n6. Elementary calculations (transformation='rdiff')")
    gdp_rdiff = nasdaqdatalink.get("FRED/GDP", transformation="rdiff")
    print(gdp_rdiff.head())


    print("\n=== Datatables Examples ===")
    # Datatables are used for non-time-series data. Look at each data product's documentation to determine format.
    # Note: Accessing premium Datatables like ZACKS/FC requires a subscription and valid API key.

    # 1. Make a Basic Data Call
    # zacks_data = nasdaqdatalink.get_table('ZACKS/FC', ticker='AAPL')
    # print(zacks_data.head())

    # 2. Set Pagination
    # Turn on pagination to return data page by page and to avoid exceeding call limits.
    # data_paginated = nasdaqdatalink.get_table('ZACKS/FC', paginate=True)

    # 3. Slice and Dice the Data
    # Request specific columns
    # data_cols = nasdaqdatalink.get_table('ZACKS/FC', paginate=True, ticker='AAPL', qopts={'columns': ['ticker', 'per_end_date']})
    
    # Filter based on column
    # data_filtered = nasdaqdatalink.get_table('ZACKS/FC', paginate=True, ticker=['AAPL', 'MSFT'], per_end_date={'gte': '2015-01-01'}, qopts={'columns':['ticker', 'per_end_date']})

if __name__ == '__main__':
    main()
