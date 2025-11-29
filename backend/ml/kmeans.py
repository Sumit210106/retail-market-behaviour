import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def prepare_customer_features(raw_df):
    df = raw_df.dropna(subset=['CustomerID'])

    customer = df.groupby('CustomerID').agg(
        total_spend=("Quantity", lambda x: (x * df.loc[x.index, "UnitPrice"]).sum()),
        total_items=("Quantity", "sum"),
        total_orders=("InvoiceNo", "nunique")
    ).reset_index()

    return customer


def run_kmeans(df, k=3):
    customer_df = prepare_customer_features(df)

    features = customer_df[["total_spend", "total_items", "total_orders"]]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=k, random_state=42)
    customer_df["cluster"] = kmeans.fit_predict(scaled_features)

    summary = []
    for i in range(k):
        group = customer_df[customer_df["cluster"] == i]

        summary.append({
            "cluster_id": i,
            "customers": len(group),
            "avg_spend": round(group["total_spend"].mean(), 2),
            "avg_items": round(group["total_items"].mean(), 2),
            "avg_orders": round(group["total_orders"].mean(), 2),
        })

    return summary
