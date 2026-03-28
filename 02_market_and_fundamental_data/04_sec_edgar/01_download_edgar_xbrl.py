# %% [markdown]
# # Download FS & Notes Data from SEC's EDGAR service

# https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets
# https://www.sec.gov/data-research/sec-markets-data/financial-statement-notes-data-sets# %%
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import date
from io import BytesIO
from zipfile import ZipFile, BadZipFile
from tqdm import tqdm
import requests
import pandas as pd

# %%
# store data in this directory since we won't use it in other chapters
data_path = Path('data')
if not data_path.exists():
    data_path.mkdir()

# %% [markdown]
# ## Download FS & Notes Data

# %%
SEC_URL = 'https://www.sec.gov/'
FSN_PATH = 'files/dera/data/financial-statement-notes-data-sets/'

# %%
# Define date range parameters for downloading data
START_DATE = '2009'
END_DATE = '2026-12-31'

# %%
filing_periods = [(d.year, d.quarter) for d in pd.date_range(START_DATE, END_DATE, freq='Q')]
filing_periods

# %%
for yr, qtr in tqdm(filing_periods):
    # set (and create) directory
    path = data_path / f'{yr}_{qtr}' / 'source'
    
    # Check if files exist to avoid re-downloading
    if path.exists() and any(path.iterdir()):
        print(f'Skipping {yr} Q{qtr}, files already exist in {path}')
        continue

    path.mkdir(parents=True, exist_ok=True)
    
    # Check if quarterly consolidated file is available first
    filing_q = f'{yr}q{qtr}_notes.zip'
    url_q = SEC_URL + FSN_PATH + filing_q
    headers = {'User-Agent': 'MachineLearningForTrading_Student admin@ml4t.com'}
    
    # Use extremely fast HEAD request to check for file existence
    if requests.head(url_q, headers=headers).status_code == 200:
        filings = [filing_q]
    else:
        # Fallback to monthly files if quarterly is not available (common for the most recent year)
        months = [qtr * 3 - 2, qtr * 3 - 1, qtr * 3]
        filings = [f'{yr}_{m:02d}_notes.zip' for m in months]
        
    for filing in filings:
        url = SEC_URL + FSN_PATH + filing
        response = requests.get(url, headers=headers)
        
        if response.status_code != 200:
            print(f'\nFile not found (status {response.status_code}): {filing}\n')
            continue
            
        # decompress and save
        try:
            with ZipFile(BytesIO(response.content)) as zip_file:
                for file in zip_file.namelist():
                    local_file = path / file
                    is_new = not local_file.exists()
                    
                    # Ensure previous file ended with newline when concatenating to avoid corrupted rows
                    if not is_new:
                        with local_file.open('rb') as f:
                            f.seek(-1, 2)
                            last_char = f.read(1)
                        if last_char != b'\n':
                            with local_file.open('ab') as f:
                                f.write(b'\n')
                                
                    # Append mode if file already exists (combining monthly files)
                    with local_file.open('ab' if not is_new else 'wb') as output:
                        lines = zip_file.open(file).readlines()
                        # Skip the header row for subsequent monthly files appended to the same TSV
                        if not is_new and len(lines) > 0:
                            lines = lines[1:]  
                        for line in lines:
                            output.write(line)
        except BadZipFile:
            print(f'\nBad zip file: {filing}\n')
            continue

# %% [markdown]
# ## Save to parquet

# %% [markdown]
# The data is fairly large and to enable faster access than the original text files permit, it is better to convert the text files to binary, columnar parquet format (see Section 'Efficient data storage with pandas' in chapter 2 for a performance comparison of various data-storage options compatible with pandas DataFrames):

# %% [markdown]
# > Some fo the `txt.tsv` source files contain a small number of faulty lines; the code below drops those lines but indicates the line numbers where you can find the errors if you would like to investigate further. 

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
