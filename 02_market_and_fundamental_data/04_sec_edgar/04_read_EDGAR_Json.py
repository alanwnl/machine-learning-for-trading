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

# ==========================================
# 1. GENERAL US-GAAP DATA POINT COUNTING
# ==========================================
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

csv_path = "data/tsla_us_gaap_counts.csv"
df_us_gaap.to_csv(csv_path, index=False)
print(f"\nSaved the full US-GAAP units overview to {csv_path}")

# ==========================================
# 2. SPECIFIC STOCK SPLIT DETECTION
# ==========================================
print("\n--- Stock Split Detection ---")
split_concept = "StockholdersEquityNoteStockSplitConversionRatio1"

if split_concept in us_gaap_data:
    split_details = us_gaap_data[split_concept]
    units = split_details.get("units", {})
    
    unique_splits = set() # Using a set to deduplicate multiple filings reporting the same event
    
    for unit_name, data_points in units.items():
        for point in data_points:
            # Look for an "instant" in time: has an 'end' date, but NO 'start' date
            if "end" in point and "start" not in point:
                date = point["end"]
                ratio = point["val"]
                unique_splits.add((date, ratio))
    
    if unique_splits:
        # Sort chronologically by date
        sorted_splits = sorted(list(unique_splits), key=lambda x: x[0])
        for split_date, ratio in sorted_splits:
            print(f"[*] Detected {int(ratio)}-for-1 Stock Split declared on: {split_date}")
    else:
        print("Stock split concept found, but no exact declaration dates were detected.")
else:
    print(f"Concept '{split_concept}' not found in the dataset.")