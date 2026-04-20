from fastapi import FastAPI
from utils.data_loader import load_default_data
from utils.data_loader import load_5lakh_data
from ml.timeseries import peak_sales_insights
from ml.knn import build_knn, recommend
from ml.kmeans import run_kmeans
from ml.decision_tree import run_decision_tree
from ml.pca import run_pca
import json
import gzip
import os
import time

from ai.agent import run_agent
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# REQUEST MODEL
# -------------------------
class QueryRequest(BaseModel):
    query: str


# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {"status": "API is running!"}


# -------------------------
# LOAD APRIORI 
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APR_PATH = os.path.join(BASE_DIR, "cached", "apriori_output.json.gz")

APRIORI_DATA = {}

try:
    print("[INFO] Loading Apriori data:", APR_PATH)
    with gzip.open(APR_PATH, "rt", encoding="utf-8") as f:
        APRIORI_DATA = json.load(f)
except Exception as e:
    print("[WARNING] Failed to load Apriori data:", str(e))


# -------------------------
# API ENDPOINTS
# -------------------------
@app.get("/apriori")
def get_apriori_results():
    return APRIORI_DATA


@app.get("/default-data")
def get_default_data():
    df = load_default_data()
    return {
        "rows": df.shape[0],
        "columns": df.columns.tolist(),
        "preview": df.head(5).to_dict(orient="records"),
    }


@app.get("/peak-sales")
def get_peak_sales():
    df = load_default_data()
    return peak_sales_insights(df)


# -------------------------
# KNN SETUP
# -------------------------
products = []
knn_model = None
sparse_matrix = None


@app.on_event("startup")
def prepare_knn():
    global products, knn_model, sparse_matrix
    try:
        df = load_default_data()
        products, knn_model, sparse_matrix = build_knn(df)
        print("KNN model loaded with", len(products), "products")
    except Exception as e:
        print("[ERROR] KNN loading failed:", str(e))


# -------------------------
# KNN ENDPOINTS
# -------------------------
@app.get("/similar-products")
def similar(product: str):
    return recommend(product, products, knn_model, sparse_matrix)


@app.get("/similar-products/all")
def all_similar():
    result = {}
    limit = 50

    for product in products[:limit]:
        result[product] = recommend(product, products, knn_model, sparse_matrix, top_k=5)

    return result


# -------------------------
# ML ENDPOINTS
# -------------------------
@app.get("/customer-segmentation")
def customer_segmentation(k: int = 3):
    df = load_default_data()
    return run_kmeans(df, k)


@app.get("/customer-spend-prediction")
def spend_prediction():
    df = load_default_data()
    return run_decision_tree(df)


@app.get("/pca-visualization")
def pca_visualization():
    df = load_default_data()
    return run_pca(df)


@app.get("/customer-behavior")
def customer_behavior():
    df = load_default_data()
    return run_pca(df)



@app.post("/ai-agent")
def ai_agent(request: QueryRequest):
    try:
        start_time = time.time()

        response = run_agent(request.query)

        end_time = time.time()

        return {
            "status": "success",
            "query": request.query,
            "response": response,
            "time_taken": round(end_time - start_time, 2)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }