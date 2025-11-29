import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


def prepare_customer_features(raw_df):
    df = raw_df.dropna(subset=["CustomerID"]).copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    customer = df.groupby("CustomerID").agg(
        total_spend=("Amount", "sum"),
        total_items=("Quantity", "sum"),
        total_orders=("InvoiceNo", "nunique"),   
        avg_order_value=("Amount", "mean"),
        last_purchase=("InvoiceDate", "max")
    ).reset_index()

    max_date = df["InvoiceDate"].max()
    customer["recency"] = (max_date - customer["last_purchase"]).dt.days

    df_recent = df[df["InvoiceDate"] >= max_date - pd.Timedelta(days=30)]
    recent_spend = df_recent.groupby("CustomerID")["Amount"].sum()
    customer["next_month_spend"] = customer["CustomerID"].map(recent_spend).fillna(0)

    return customer


def run_decision_tree(df):
    customer = prepare_customer_features(df)

    X = customer[["total_spend", "total_items", "total_orders", "avg_order_value", "recency"]]
    Y = customer["next_month_spend"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X)

    result = []
    for cust_id, pred in zip(customer["CustomerID"], predictions):
        result.append({
            "CustomerID": int(cust_id),
            "predicted_spend": round(float(pred), 2)
        })

    return result
