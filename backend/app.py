from fastapi import FastAPI
from utils.data_loader import load_default_data
from utils.data_loader import load_5lakh_data
from ml.timeseries import peak_sales_insights
from ml.knn import build_knn, recommend
# from ml.apriori import run_apriori
from ml.kmeans import run_kmeans
from ml.decision_tree import run_decision_tree
from ml.pca import run_pca
import json
import gzip
import os
# cors
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# HOME
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"status": "API is running!"}


# --------------------------------------------------------
# Load Apriori output once when the server starts
# --------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APR_PATH = os.path.join(BASE_DIR, "cached", "apriori_output.json.gz")

print("[INFO] Loading Apriori data:", APR_PATH)

with gzip.open(APR_PATH, "rt", encoding="utf-8") as f:
    APRIORI_DATA = json.load(f)


# --------------------------------------------------------
# API ENDPOINT
# --------------------------------------------------------

@app.get("/apriori")
def get_apriori_results():
    return APRIORI_DATA

# ---------------------------------------------------------
# DEFAULT DATA PREVIEW
# ---------------------------------------------------------
@app.get("/default-data")
def get_default_data():
    df = load_default_data()
    return {
        "rows": df.shape[0],
        "columns": df.columns.tolist(),
        "preview": df.head(5).to_dict(orient="records"),
    }


# ---------------------------------------------------------
# PEAK SALES INSIGHTS
# ---------------------------------------------------------
@app.get("/peak-sales")
def get_peak_sales():
    df = load_default_data()
    return peak_sales_insights(df)


# ---------------------------------------------------------
# KNN SIMILAR PRODUCTS (Startup Cache)
# ---------------------------------------------------------
products = []
products = []
knn_model = None
sparse_matrix = None


@app.on_event("startup")
def prepare_knn():
    global products, knn_model, sparse_matrix
    df = load_default_data()
    products, knn_model, sparse_matrix = build_knn(df)
    print("KNN model loaded with", len(products), "products")


# ---------------------------------------------------------
# KNN ENDPOINTS
# ---------------------------------------------------------
@app.get("/similar-products")
def similar(product: str):
    return recommend(product, products, knn_model, sparse_matrix)

# Here limit to first 50 .
@app.get("/similar-products/all")
def all_similar():
    result = {}
    limit = 50  # avoid API timeout

    for product in products[:limit]:
        result[product] = recommend(product, products, knn_model, sparse_matrix, top_k=5)

    return result


# ---------------------------------------------------------
# APRIORI – Frequently Bought Together
# ---------------------------------------------------------
# @app.get("/frequently-bought-together")
# def get_fbt(
#     min_support: float = 0.001,
#     min_confidence: float = 0.01,
#     top_k: int = 20
# ):
#     df =load_5lakh_data()

#     result = run_apriori(df, min_support, min_confidence)
    
#     # run_apriori returns {"message": "...", "rules": [...]}
#     rules = result.get("rules", [])
    
#     # Sort by combination_size (descending) first, then by lift (descending)
#     # This prioritizes showing 3-4 item combinations before 2-item ones
#     sorted_rules = sorted(
#         rules, 
#         key=lambda x: (x.get("combination_size", 0), x["lift"]), 
#         reverse=True
#     )
    
#     return {
#         "message": result.get("message", "Success"),
#         "rules": sorted_rules[:top_k]
#     }



# ---------------------------
# KMEANS ENDPOINT
# ---------------------------
@app.get("/customer-segmentation")
def customer_segmentation(k: int = 3):
    """
    Run K-Means clustering for customer segmentation.
    Returns cluster summaries.
    """
    df = load_default_data()
    result = run_kmeans(df, k)
    return result

# ---------------------------
# DECISION TREE ENDPOINT
# ---------------------------

@app.get("/customer-spend-prediction")
def spend_prediction():
    df = load_default_data()
    result = run_decision_tree(df)
    return result


# ---------------------------
# PCA ENDPOINT
# ---------------------------
@app.get("/pca-visualization")
def pca_visualization():
    """
    Run PCA for customer visualization.
    Returns 2D coordinates for each customer.
    """
    df = load_default_data()
    result = run_pca(df)
    return result


# ---------------------------
# PCA
# ---------------------------

@app.get("/customer-behavior")
def customer_behavior():
    df = load_default_data()
    result = run_pca(df)
    return result