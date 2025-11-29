import pandas as pd

# Your exact LOCAL file path
LOCAL_FILE = "/Users/sumitnayak/Desktop/retail market behaviour/backend/data/online_retail.xlsx"

# Number of rows you want to sample
SAMPLE_SIZE = 16000

# Output filename
OUTPUT_FILE = "uci_retail_1600_rows.xlsx"

print("Reading LOCAL UCI dataset...")

# Read Excel file
df = pd.read_excel(LOCAL_FILE, engine="openpyxl")

print("Cleaning dataset...")
df = df.dropna(how="all")  # remove completely empty rows

# Random sampling
print(f"Sampling {SAMPLE_SIZE} random rows...")
sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)

# Save to Excel
sample_df.to_excel(OUTPUT_FILE, index=False)

print(f"\nSUCCESS: Saved 1600-row sample to {OUTPUT_FILE}")
