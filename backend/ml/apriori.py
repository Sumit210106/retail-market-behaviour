import pandas as pd
# from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(
    df,
    min_support=0.0005,      # Very low to find rare 4-item patterns
    min_confidence=0.1,
    top_n_products=120,      # Balanced for speed and pattern quality
    max_len=4,               # Allow up to 4-item combinations
    max_rules=50             # Increased to get more results
):

    """
    Highly optimized Apriori for large datasets.
    Supports:
    - 2-itemsets
    - 3-itemsets
    - 4-itemsets (adjustable with max_len)

    Returns strong association rules in JSON format.
    """

    # ---------------------------------------------------
    # STEP 1: Clean dataset
    # ---------------------------------------------------
    df = df[['InvoiceNo', 'Description', 'Quantity']].dropna()

    # Remove cancelled invoices + negative quantity
    df = df[df["Quantity"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Select top N most frequent products (boosts speed x10)
    product_counts = df["Description"].value_counts()
    top_products = product_counts.head(top_n_products).index
    df = df[df["Description"].isin(top_products)]

    print(
        f"Apriori: {df['InvoiceNo'].nunique()} transactions, "
        f"{len(top_products)} products, max_len={max_len}"
    )

    # ---------------------------------------------------
    # STEP 2: Build basket matrix (Boolean One-Hot)
    # ---------------------------------------------------
    basket = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

    basket = (basket > 0).astype(bool)

    # ---------------------------------------------------
    # STEP 3: Frequent Itemsets
    # ---------------------------------------------------
    frequent_items = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len
    )

    if frequent_items.empty:
        return {"message": "No frequent itemsets found.", "rules": []}

    print("Frequent itemsets found:", frequent_items.shape[0])

    # ---------------------------------------------------
    # STEP 4: Build strong rules
    # ---------------------------------------------------
    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return {"message": "No rules found.", "rules": []}

    # Keep useful columns
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]

    # Convert frozensets → lists
    rules["antecedents"] = rules["antecedents"].apply(list)
    rules["consequents"] = rules["consequents"].apply(list)

    # Sort by strength
    rules = rules.sort_values(by="lift", ascending=False)

    # ---------------------------------------------------
    # STEP 5: Format JSON output
    # ---------------------------------------------------
    formatted = []
    for _, row in rules.head(max_rules).iterrows():

        formatted.append({
            "buy_together": row["antecedents"],
            "recommend": row["consequents"],
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 3),
            "combination_size": len(row["antecedents"]) + len(row["consequents"])
        })

    return {
        "message": f"{len(formatted)} rules found.",
        "rules": formatted
    }
