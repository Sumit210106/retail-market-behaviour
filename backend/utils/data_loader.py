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


cached_df_5lakh = None

def load_5lakh_data(nrows=10000):
    """
    Load the large retail dataset.
    By default, loads first 10k rows for faster response time.
    Set nrows=None to load all data (may cause memory issues).
    """
    global cached_df_5lakh

    if cached_df_5lakh is None:
        print(f"Loading retail data (nrows={nrows})...")
        cached_df_5lakh = pd.read_excel(
            "data/online_retail.xlsx",
            engine="openpyxl",
            nrows=nrows
        )
        print(f"Loaded {len(cached_df_5lakh)} rows")

    return cached_df_5lakh

