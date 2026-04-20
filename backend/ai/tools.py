import json
import gzip
import os

from utils.data_loader import load_default_data
from ml.kmeans import run_kmeans
from ml.timeseries import peak_sales_insights


df = load_default_data()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APR_PATH = os.path.join(BASE_DIR, "cached", "apriori_output.json.gz")

with gzip.open(APR_PATH, "rt", encoding="utf-8") as f:
    APRIORI_DATA = json.load(f)


def segmentation_tool(input_str: str = ""):
    try:
        result = run_kmeans(df, k=3)
        return json.dumps(result)
    except Exception as e:
        return f"Error in segmentation tool: {str(e)}"


def peak_sales_tool(input_str: str = ""):
    try:
        result = peak_sales_insights(df)
        return json.dumps(result)
    except Exception as e:
        return f"Error in peak sales tool: {str(e)}"


def basket_tool(input_str: str = ""):
    try:
        rules = APRIORI_DATA.get("rules", [])
        return json.dumps(rules[:50])
    except Exception as e:
        return f"Error in basket tool: {str(e)}"



def get_tools():
    return [
        segmentation_tool,
        peak_sales_tool,
        basket_tool
    ]