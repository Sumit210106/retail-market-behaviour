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
