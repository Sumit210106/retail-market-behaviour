import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

def build_knn(df):
    products = df["Description"].astype(str).tolist()

    vec = CountVectorizer()
    sparse_matrix = vec.fit_transform(products)

    # Use NearestNeighbors with cosine metric
    # n_neighbors=20 is usually enough for top-k recommendations
    model = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
    model.fit(sparse_matrix)

    return products, model, sparse_matrix


def recommend(product_name, products, model, sparse_matrix, top_k=5):
    if product_name not in products:
        return []

    idx = products.index(product_name)
    
    # Get the vector for the query product
    query_vec = sparse_matrix[idx]

    # Find neighbors
    # kneighbors returns (distances, indices)
    # distance in cosine metric is 1 - similarity, so smaller is better.
    # Fetch more candidates to account for duplicates
    distances, indices = model.kneighbors(query_vec, n_neighbors=min(top_k * 5, len(products)))
    
    # Flatten results
    distances = distances.flatten()
    indices = indices.flatten()

    recommendations = []
    seen = set()
    seen.add(product_name)

    # Skip the first one because it's the product itself (distance ~ 0)
    # Iterate through more neighbors to find distinct ones
    for i in range(1, len(indices)):
        neighbor_idx = indices[i]
        name = products[neighbor_idx]
        
        if name in seen:
            continue
        seen.add(name)

        distance = distances[i]
        similarity = 1 - distance  # Convert distance back to similarity for display

        recommendations.append({
            "product": name,
            "similarity": round(float(similarity), 3)
        })
        
        if len(recommendations) >= top_k:
            break

    return recommendations
