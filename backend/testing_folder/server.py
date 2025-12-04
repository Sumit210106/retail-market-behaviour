from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from analysis import run_analysis, sales_by_time , segment_customers_by_basket

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------
# question - 1
# ---------------------

@app.get("/analysis")
def analysis(min_support: float, min_confidence: float):
    rules = run_analysis(min_support, min_confidence)
    return {"rules": rules}

# api -> <http://127.0.0.1:8000>/analysis?min_support=0.02&min_confidence=0.2





# ---------------------
# question - 2
# ---------------------

@app.get("/sales-by-time")
def sales_by_time_endpoint():
    return sales_by_time()

# api -> http://127.0.0.1:8000/sales-by-time





# ---------------------
# question - 3
# ---------------------

@app.get("/customer-segments")
def customer_segments():
    return segment_customers_by_basket()

# api -> http://127.0.0.1:8000/customer-segments
