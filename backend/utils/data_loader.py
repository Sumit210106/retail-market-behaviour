import pandas as pd

DEFAULT_URL = "https://archive.ics.uci.edu/static/public/352/data.csv"



LOCAL_DATA_PATH = "data/uci_retail_1600_rows.xlsx"


cached_df = None

def load_default_data():
    global cached_df

    if cached_df is None:
        cached_df = pd.read_excel(
            LOCAL_DATA_PATH,
            engine="openpyxl"
        )

    return cached_df


import os

cached_df_5lakh = None

def load_5lakh_data(nrows=10000):
    """
    Load the large retail dataset.
    Prioritizes CSV if available, otherwise falls back to Excel.
    """
    global cached_df_5lakh

    if cached_df_5lakh is None:
        csv_path = "data/online_retail.csv"
        excel_path = "data/online_retail.xlsx"
        
        if os.path.exists(csv_path):
            print(f"Loading retail data from CSV (nrows={nrows})...")
            cached_df_5lakh = pd.read_csv(csv_path, nrows=nrows)
        else:
            print(f"Loading retail data from Excel (nrows={nrows})...")
            cached_df_5lakh = pd.read_excel(
                excel_path,
                engine="openpyxl",
                nrows=nrows
            )
        
        print(f"Loaded {len(cached_df_5lakh)} rows")

    return cached_df_5lakh

