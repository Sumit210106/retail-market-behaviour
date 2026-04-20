import json
import gzip
import os

from langchain.tools import Tool

from utils.data_loader import load_default_data
from ml.kmeans import run_kmeans
from ml.timeseries import peak_sales_insights


df = load_default_data()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APR_PATH = os.path.join(BASE_DIR, "cached", "apriori_output.json.gz")

with gzip.open(APR_PATH, "rt", encoding="utf-8") as f:
    APRIORI_DATA = json.load(f)


def segmentation_tool(_):
    return json.dumps(run_kmeans(df, k=3))

def peak_sales_tool(_):
    return json.dumps(peak_sales_insights(df))

def basket_tool(_):
    return json.dumps(APRIORI_DATA["rules"][:50])

def get_tools():
    return [
        Tool(
            name="Customer Segmentation",
            func=segmentation_tool,
            description="Use this to analyze customer groups and identify high-value segments"
        ),
        Tool(
            name="Peak Sales Analysis",
            func=peak_sales_tool,
            description="Use this to understand best sales time and trends"
        ),
        Tool(
            name="Market Basket Analysis",
            func=basket_tool,
            description="Use this to find frequently bought together products"
        )
    ]