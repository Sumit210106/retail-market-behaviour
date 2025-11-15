import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.join(BASE_DIR, "data", "online_retail.xlsx")

df = pd.read_excel(EXCEL_PATH)

print("Dataset loaded into shared memory:", df.shape)
