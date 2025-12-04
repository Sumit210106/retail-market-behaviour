import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def prepare_customer_features(raw_df):
    """
    Aggregates raw transaction data into customer-level features.
    Same logic as in kmeans.py to ensure consistency.
    """
    # Drop rows with missing CustomerID
    df = raw_df.dropna(subset=['CustomerID'])

    # Aggregate by CustomerID
    customer = df.groupby('CustomerID').agg(
        total_spend=("Quantity", lambda x: (x * df.loc[x.index, "UnitPrice"]).sum()),
        total_items=("Quantity", "sum"),
        total_orders=("InvoiceNo", "nunique")
    ).reset_index()

    return customer

def run_pca(df, n_components=2):
    """
    Runs PCA on customer data to reduce dimensions for visualization.
    Returns a list of dictionaries containing customer ID, original metrics, and PCA coordinates.
    """
    customer_df = prepare_customer_features(df)
    
    # Select features for PCA
    features = customer_df[["total_spend", "total_items", "total_orders"]]
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(scaled_features)
    
    # Add PCA coordinates to the dataframe
    customer_df['x'] = pca_result[:, 0]
    customer_df['y'] = pca_result[:, 1]
    
    # Convert to list of dictionaries for JSON response
    # Rounding float values for cleaner output
    results = []
    for _, row in customer_df.iterrows():
        results.append({
            "customer_id": int(row["CustomerID"]),
            "total_spend": round(row["total_spend"], 2),
            "total_items": int(row["total_items"]),
            "total_orders": int(row["total_orders"]),
            "x": round(row["x"], 4),
            "y": round(row["y"], 4)
        })
        
    return {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "data": results
    }
