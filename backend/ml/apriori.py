import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def run_apriori(
    df,
    min_support=0.005,        # smart default
    min_confidence=0.1,       # smart default
    max_rules=20
):
    """
    Runs Apriori on the retail dataset and returns strong association rules.

    OUTPUT FORMAT (list of dicts):
    [
        {
            "buy_together": ["A", "B"],
            "recommend": ["C"],
            "support": 0.0124,
            "confidence": 0.62,
            "lift": 2.14
        },
        ...
    ]
    """

    # ---------------------------------------------------
    # STEP 1: Clean dataset
    # ---------------------------------------------------
    df = df[['InvoiceNo', 'Description', 'Quantity']].dropna()

    # Group by invoice, convert to basket matrix
    basket = (
        df.groupby(['InvoiceNo', 'Description'])['Quantity']
        .sum()
        .unstack()
        .fillna(0)
    )

    # Convert quantities to 0/1 (presence)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # ---------------------------------------------------
    # STEP 2: Frequent itemsets
    # ---------------------------------------------------
    frequent_items = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        max_len=3
    )

    if frequent_items.empty:
        return {
            "message": "No frequent patterns found.",
            "rules": []
        }

    # ---------------------------------------------------
    # STEP 3: Build rules
    # ---------------------------------------------------
    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return {
            "message": "No association rules found.",
            "rules": []
        }

    # Keep useful columns only
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    # Convert frozensets to lists
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x))

    # Sort strongest → weakest
    rules = rules.sort_values(by='lift', ascending=False)

    # ---------------------------------------------------
    # STEP 4: Convert to JSON-friendly format
    # ---------------------------------------------------
    formatted_rules = []
    for _, row in rules.head(max_rules).iterrows():
        formatted_rules.append({
            "buy_together": row['antecedents'],
            "recommend": row['consequents'],
            "support": round(float(row['support']), 4),
            "confidence": round(float(row['confidence']), 4),
            "lift": round(float(row['lift']), 3),
        })

    # ---------------------------------------------------
    # Final output
    # ---------------------------------------------------
    return {
        "message": f"{len(formatted_rules)} rules found.",
        "rules": formatted_rules
    }
