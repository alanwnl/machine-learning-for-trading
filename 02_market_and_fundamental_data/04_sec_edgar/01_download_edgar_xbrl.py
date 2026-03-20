# %% [markdown]
# # Download FS & Notes Data from SEC's EDGAR service

# %%
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
START_DATE = '2014'
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
    
    # define url and get file
    filing = f'{yr}q{qtr}_notes.zip'
    url = SEC_URL + FSN_PATH + filing
    headers = {'User-Agent': 'MachineLearningForTrading_Student admin@ml4t.com'}
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
            df = pd.read_csv(f, sep='\t', encoding='latin1', low_memory=False, error_bad_lines=False)
            df.to_parquet(parquet_path / file_name)
        except Exception as e:
            print(e, ' | ', f)
        # optional: uncomment to delete original .tsv
#         else:
            # f.unlink
