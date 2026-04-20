import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

def build_knn(df):
    products = df["Description"].astype(str).tolist()

    vec = CountVectorizer()
    sparse_matrix = vec.fit_transform(products)

    model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    model.fit(sparse_matrix)

    return products, model, sparse_matrix


def recommend(product_name, products, model, sparse_matrix, top_k=5):
    if product_name not in products:
        return []

    idx = products.index(product_name)
    
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

        distance = distances[i]
        similarity = 1 - distance  

        recommendations.append({
            "product": name,
            "similarity": round(float(similarity), 3)
        })
        
        if len(recommendations) >= top_k:
            break

    return recommendations
