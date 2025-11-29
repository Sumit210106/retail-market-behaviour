import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_knn(df):
    products = df["Description"].astype(str).tolist()

    vec = CountVectorizer()
    matrix = vec.fit_transform(products)

    sim_matrix = cosine_similarity(matrix)

    return products, sim_matrix


def recommend(product_name, products, sim_matrix, top_k=5):
    if product_name not in products:
        return []

    idx = products.index(product_name)
    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in scores[1 : top_k + 1]:
        recommendations.append({
            "product": products[i],
            "similarity": round(float(score), 3)
        })

    return recommendations
