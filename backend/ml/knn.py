import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors


def build_knn(df):
    products = (
        df["Description"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    vec = CountVectorizer()
    sparse_matrix = vec.fit_transform(products)

    model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    model.fit(sparse_matrix)

    return products, model, sparse_matrix


def recommend(product_name, products, model, sparse_matrix, top_k=5):
    if not products or model is None or sparse_matrix is None:
        return []

    product_query = str(product_name).strip().upper()

    if product_query not in products:
        found = False
        for p in products:
            if product_query in p:
                product_query = p
                found = True
                break
        if not found:
            return []

    idx = products.index(product_query)
    query_vec = sparse_matrix[idx]
    distances, indices = model.kneighbors(query_vec, n_neighbors=min(top_k * 5, len(products)))

    distances = distances.flatten()
    indices = indices.flatten()

    recommendations = []
    seen = set()
    seen.add(product_name)

    for i in range(1, len(indices)):
        neighbor_idx = indices[i]
        name = products[neighbor_idx]

        if name in seen:
            continue
        seen.add(name)

        similarity = round(float(1 - distances[i]), 3)

        recommendations.append({
            "product": name,
            "similarity": similarity
        })
        if len(recommendations) >= top_k:
            break

    return recommendations
