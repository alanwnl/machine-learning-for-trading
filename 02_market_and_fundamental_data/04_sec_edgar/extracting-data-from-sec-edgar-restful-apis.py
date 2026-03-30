# %% [markdown]
# # Using SEC EDGAR RESTful data APIs
# 
# This notebook shows how to retrieve information reported by regulated entities to U.S. Securities and Exchange Commision (SEC).
# 
# SEC is maintainig EDGAR system with information about all regulated enties (companies, funds, individuals). Accessing the data is free and there is number of [various ways how to access the data](https://www.sec.gov/os/accessing-edgar-data).
# 
# "data.sec.gov" was created to host RESTful data Application Programming Interfaces (APIs) delivering JSON-formatted data to external customers and to web pages on SEC.gov. These APIs do not require any authentication or API keys to access.
# 
# Currently included in the APIs are the submissions history by filer and the XBRL data from financial statements (forms 10-Q, 10-K,8-K, 20-F, 40-F, 6-K, and their variants).
# 
# The JSON structures are updated throughout the day, in real time, as submissions are disseminated.

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
# data are published in JSON format so we will need json library
import json


# %% [markdown]
# # Finding CIK of company
# 
# EDGAR assigns to filers a unique numerical identifier, known as a Central Index Key (CIK), when they sign up to make filings to the SEC. CIK numbers remain unique to the filer; they are not recycled. 
# 
# List of all CIKs matched with entity name is available for download [(13 MB, text file)](https://www.sec.gov/Archives/edgar/cik-lookup-data.txt). Note that this list includes funds and individuals and is historically cumulative for company names. Thus a given CIK may be associated with multiple names in the case of company or fund name changes, and the list contains some entities that no longer file with the SEC.
# 
# We will be using smaller (611 kB) JSON [kaggle dataset](https://www.kaggle.com/datasets/svendaj/sec-edgar-cik-ticker-exchange), which is sourcing data directly at EDGAR and is input for this notebook. This dataset contains only companies names, CIK, ticker and associated stock exchange.

# %%
# # Let's convert CIK JSON to pandas DataFrame
# # First load the data into python dictionary
# https://www.sec.gov/files/company_tickers_exchange.json
# with open("../input/sec-edgar-cik-ticker-exchange/company_tickers_exchange.json", "r") as f:
#     CIK_dict = json.load(f)

import requests

url = "https://www.sec.gov/files/company_tickers_exchange.json"

# The SEC requires you to identify yourself in the User-Agent header
headers = {
    "User-Agent": "Your Name YourEmail@example.com" 
}

# Fetch the data from the URL
response = requests.get(url, headers=headers)

# Check if the request was successful (Status code 200)
if response.status_code == 200:
    # .json() automatically parses the JSON text into a Python dictionary
    CIK_dict = response.json() 
    print("Successfully fetched the data!")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")

# %%
# dataset contains two sections
CIK_dict.keys()

# %%
# fields is specifying meaning and and order of company data
# we will use it as columns names
CIK_dict["fields"]

# %%
# data section is list of records/lists for each company
# we will use it as DataFrame rows
print("Number of company records:", len(CIK_dict["data"]))
CIK_dict["data"][:5]    # first 5 records

# %%
# convert CIK_dict to pandas
CIK_df = pd.DataFrame(CIK_dict["data"], columns=CIK_dict["fields"])
CIK_df

# %% [markdown]
# ## Select the ticker of company used in this example
# 
# Subsequent information retrieval will be using selected `ticker` and associated CIK

# %%
# finding company row with given ticker
ticker = "TSLA"
CIK_df[CIK_df["ticker"] == ticker]

# %%
CIK = CIK_df[CIK_df["ticker"] == ticker].cik.values[0]

# %%
# finding companies containing substring in company name
substring = "oil"
CIK_df[CIK_df["name"].str.contains(substring, case=False)]

# %% [markdown]
# # Entity’s current filing history
# 
# Each entity’s current filing history is available at the following URL:
# 
# * https://data.sec.gov/submissions/CIK##########.json
# 
# Where the ########## is the entity’s 10-digit Central Index Key (CIK), including leading zeros.
# 
# This JSON data structure contains metadata such as current name, former name, and stock exchanges and ticker symbols of publicly-traded companies. The object’s property path contains at least one year’s of filing or to 1,000 (whichever is more) of the most recent filings in a compact columnar data array. If the entity has additional filings, files will contain an array of additional JSON files and the date range for the filings each one contains.

# %%
# preparation of input data, using ticker and CIK set earlier
url = f"https://data.sec.gov/submissions/CIK{str(CIK).zfill(10)}.json"
url

# %% [markdown]
# # Reading from RESTful API
# 
# EDGAR requires that HTTP requests will be identified with proper [UserAgent in header and comply with fair use policy (currently max. 10 requests per second)](https://www.sec.gov/os/accessing-edgar-data). At minimum you need to supply your own e-mail adress in User-Agent field (otherwise you will get 403/Forbiden error). If you will provide Host field, please be sure use data.sec.gov server and not www.sec.gov as mentioned in example (this would result in 404/Not Found error).

# %%
# read response from REST API with `requests` library and format it as python dict

import requests
header = {
  "User-Agent": "your.email@email.com"#, # remaining fields are optional
#    "Accept-Encoding": "gzip, deflate",
#    "Host": "data.sec.gov"
}

company_filings = requests.get(url, headers=header).json()
company_filings.keys()

# %%
company_filings["addresses"]

# %%
company_filings["filings"]["recent"].keys()

# %% [markdown]
# # Creating DataFrame with submitted filings
# 
# `company_filings["filings"]["recent"]` contains up to 1000 last submitted filings sorted from latest to oldest.

# %%
company_filings_df = pd.DataFrame(company_filings["filings"]["recent"])
company_filings_df

# %%
# filter only Annual reports
company_filings_df[company_filings_df.form == "10-K"]

# %% [markdown]
# # Accessing specific filing document
# 
# Let's download latest Annual Report (10-K). Files are stored in browsable directory structure for CIK and accession-number: 
# * https://www.sec.gov/Archives/edgar/data/{CIK}/{accession-number}/

# %%
access_number = company_filings_df[company_filings_df.form == "10-K"].accessionNumber.values[0].replace("-", "")

file_name = company_filings_df[company_filings_df.form == "10-K"].primaryDocument.values[0]

url = f"https://www.sec.gov/Archives/edgar/data/{CIK}/{access_number}/{file_name}"
url

# %%
# dowloading and saving requested document to working directory
req_content = requests.get(url, headers=header).content.decode("utf-8")

with open(file_name, "w") as f:
    f.write(req_content)

# %% [markdown]
# ## and saving it as PDF
# 

# %%
# pip install weasyprint

# %%
# import os
# import sys

# # Add Homebrew's lib directory to the fallback search path for WeasyPrint dependencies
# if sys.platform == 'darwin':
#     os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/opt/homebrew/lib:' + os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')

# from weasyprint import HTML

# HTML(string=req_content, base_url="").write_pdf(file_name + ".pdf")

# %%
# !ls -al .

# %% [markdown]
# # XBRL data APIs
# 
# Extensible Business Markup Language (XBRL) is an XML-based format for reporting financial statements used by the SEC and financial regulatory agencies across the world. XBRL, in a separate XML file or more recently embedded in quarterly and annual HTML reports as inline XBRL, was first required by the SEC in 2009. XBRL facts must be associated for a standard US-GAAP or IFRS taxonomy. Companies can also extend standard taxonomies with their own custom taxonomies.
# 
# The following XBRL APIs aggregate facts from across submissions that
# 1. Use a non-custom taxonomy (e.g. us-gaap, ifrs-full, dei, or srt)
# 1. Apply to the entire filing entity
# 
# This ensures that facts have a consistent context and meaning across companies and between filings and are comparable between companies and across time.
# 
# ## All company concepts data
# ## data.sec.gov/api/xbrl/companyfacts/
# 
# This API returns all the company concepts data for a company into a single API call:
# 
# * https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

# %%
url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{str(CIK).zfill(10)}.json"
url

# %%
company_facts = requests.get(url, headers=header).json()

# get the current assets values as reported over time and make it pandas DataFrame
curr_assets_df = pd.DataFrame(company_facts["facts"]["us-gaap"]["AssetsCurrent"]["units"]["USD"])
curr_assets_df

# %%
# get just values reported in valid frame and plot them
curr_assets_df[curr_assets_df.frame.notna()]


# %%
import plotly.express as px
pd.options.plotting.backend = "plotly" 

curr_assets_df.plot(x="end", y="val", 
                    title=f"{company_filings['name']}, {ticker}: Current Assets",
                   labels= {
                       "val": "Value ($)",
                       "end": "Quarter End"
                   })

# %% [markdown]
# ## Getting datapoints of single concept
# ## data.sec.gov/api/xbrl/companyconcept/
# 
# The company-concept API returns all the XBRL disclosures from a single company (CIK) and concept (a taxonomy and tag) into a single JSON file, with a separate array of facts for each units on measure that the company has chosen to disclose (e.g. net profits reported in U.S. dollars and in Canadian dollars).
# 
# * https://data.sec.gov/api/xbrl/companyconcept/CIK##########/us-gaap/AccountsPayableCurrent.json
# 

# %%
# let's retrieve current assets for comparision with company facts API
url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{str(CIK).zfill(10)}/us-gaap/AssetsCurrent.json"
url

# %%
curr_assets_dict = requests.get(url, headers=header).json()
curr_assets_dict.keys()

# %%
curr_assets_dict["tag"]

# %%
# first 5 datapoints
curr_assets_dict["units"]["USD"][:5]

# %%
# this should be resulting in same DataFrame as retrieved through companyfacts API and selected through taxonomy us-gaap, AssetsCurrent concept/tag and units USD
curr_assets_df = pd.DataFrame(curr_assets_dict["units"]["USD"])
curr_assets_df

# %% [markdown]
# ## Getting one fact from requested period/frame
# ## data.sec.gov/api/xbrl/frames/
# 
# The xbrl/frames API aggregates one fact for **each** reporting entity that is last filed that most closely fits the calendrical period requested. This API supports for annual, quarterly and instantaneous data:
# 
# * https://data.sec.gov/api/xbrl/frames/us-gaap/AccountsPayableCurrent/USD/CY2019Q1I.json
# 
# Where the units of measure specified in the XBRL contains a numerator and a denominator, these are separated by “-per-” such as “USD-per-shares”. Note that the default unit in XBRL is “pure”.
# 
# The period format is CY#### for annual data (duration 365 days +/- 30 days), CY####Q# for quarterly data (duration 91 days +/- 30 days), and CY####Q#I for instantaneous data. Because company financial calendars can start and end on any month or day and even change in length from quarter to quarter to according to the day of the week, the frame data is assembled by the dates that best align with a calendar quarter or year. Data users should be mindful different reporting start and end dates for facts contained in a frame.

# %%
# Let's retrieve all data about current assets in Q4 of 2021
fact = "AssetsCurrent"
year = 2021
quarter = "Q1I"

url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{fact}/USD/CY{year}{quarter}.json"
url

# %%
curr_assets_dict = requests.get(url, headers=header).json()
curr_assets_dict.keys()

# %%
# let's convert all data of requested period to pandas dataframe
curr_assets_df = pd.DataFrame(curr_assets_dict["data"])
curr_assets_df.sort_values("val", ascending=False)

# %%
company_facts["facts"].keys()

# %%
company_facts["entityName"]

# %%
CIK = 320193
url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{str(CIK).zfill(10)}/dei/EntityRegistrantName.json"
url

# %% [markdown]
# ## Retrieving Specific Tags: GrossProfit vs Custom Tags
# 
# We can retrieve `GrossProfit` via the `companyconcept` API because it is a standard `us-gaap` tag.
# 
# **Important Note:** Custom extensions like `SalesRevenueAutomotive` and `CostOfRevenuesAutomotive` that Tesla uses in its filings are **not available** through the `data.sec.gov` RESTful APIs (`companyconcept` and `companyfacts`). The SEC APIs strictly serve standardized taxonomy tags (`us-gaap`, `dei`, `ifrs-full`, etc.) and filter out company-specific custom extensions. 
# 
# To get custom tags like Automotive revenues, you have to use the bulk XBRL Parquet files (as you did previously) or parse the raw XMLs. We can, however, use this REST API method to fetch the standardized `GrossProfit`.

# %%
# Let's switch back to Tesla's CIK to get GrossProfit
cik_tsla = "0001318605"
url_gp = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik_tsla}/us-gaap/GrossProfit.json"

gp_response = requests.get(url_gp, headers=header)
if gp_response.status_code == 200:
    gp_dict = gp_response.json()
    gp_df = pd.DataFrame(gp_dict["units"]["USD"])
    
    # Filter for standard quarterly/annual frames to clean up the dataframe
    gp_df = gp_df[gp_df['frame'].notna()].copy()
    
    # Format the dates
    gp_df['end'] = pd.to_datetime(gp_df['end'])
    gp_df = gp_df.sort_values('end').reset_index(drop=True)
    
    print("Successfully fetched GrossProfit! Here are the most recent standard filings:\n")
    # Display the tail of the dataframe
    print(gp_df[['end', 'val', 'frame', 'form']].tail(10))
else:
    print(f"Failed to fetch GrossProfit data. Status: {gp_response.status_code}")

gp_df

