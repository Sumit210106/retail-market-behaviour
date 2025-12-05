import os
import json
import gzip
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# =============================================================================
# CONFIG (for ~8k invoices, 2.7k items)
# =============================================================================
MIN_SUPPORT = 0.0005        # 0.05%
MIN_CONFIDENCE = 0.05
TOP_K_RULES = 300
OUTPUT_FILENAME = "apriori_output.json.gz"


def get_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# LOAD DATA
# =============================================================================
def load_uci_retail():
    root = get_root()
    path = os.path.join(root, "data", "uci_retail_1600_rows.xlsx")

    print("[INFO] Loading UCI Retail dataset:", path)
    df = pd.read_excel(path)

    df.columns = df.columns.str.strip().str.lower()

    df = df[df["description"].notna()]
    df = df[df["quantity"] > 0]

    df = df[["invoiceno", "description"]]
    df.columns = ["InvoiceNo", "Description"]

    print("[INFO] Loaded shape:", df.shape)
    return df


# =============================================================================
# BUILD BASKET
# =============================================================================
def preprocess(df):
    print("[INFO] Building Invoice × Item matrix…")

    basket = (
        df.groupby(['InvoiceNo', 'Description'])['Description']
        .count()
        .unstack()
        .fillna(0)
    )

    basket = basket.astype(bool)

    print("[INFO] Basket shape:", basket.shape)
    return basket


# =============================================================================
# RUN FP-GROWTH + RULES
# =============================================================================
def run_mba():
    df = load_uci_retail()
    basket = preprocess(df)

    print(f"[INFO] Running FP-Growth (min_support={MIN_SUPPORT})…")
    freq_items = fpgrowth(basket, min_support=MIN_SUPPORT, use_colnames=True)

    print("[INFO] Frequent itemsets:", len(freq_items))

    # Convert frozenset → list for JSON
    freq_items["itemsets"] = freq_items["itemsets"].apply(list)

    print(f"[INFO] Generating rules (min_conf={MIN_CONFIDENCE})…")
    rules = association_rules(freq_items, metric="confidence", min_threshold=MIN_CONFIDENCE)

    print("[INFO] Rules generated:", len(rules))

    if rules.empty:
        print("⚠️ No rules found — try lowering MIN_CONFIDENCE further")

    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]

    # frozenset → list
    rules["antecedents"] = rules["antecedents"].apply(list)
    rules["consequents"] = rules["consequents"].apply(list)

    rules = rules.sort_values(by="lift", ascending=False).head(TOP_K_RULES)

    return {
        "frequent_itemsets": freq_items.to_dict(orient="records"),
        "rules": rules.to_dict(orient="records"),
        "meta": {
            "min_support": MIN_SUPPORT,
            "min_confidence": MIN_CONFIDENCE,
            "top_k_rules": TOP_K_RULES,
            "dataset_rows": len(df)
        }
    }


# =============================================================================
# SAVE .JSON.GZ
# =============================================================================
def save_output(data):
    root = get_root()
    cache_dir = os.path.join(root, "cached")
    os.makedirs(cache_dir, exist_ok=True)

    path = os.path.join(cache_dir, OUTPUT_FILENAME)
    print("[INFO] Saving to:", path)

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("[INFO] Saved ✔")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("      Market Basket Analysis (UCI Retail)         ")
    print("==================================================")

    results = run_mba()
    save_output(results)

    print("\nDONE ✔")
