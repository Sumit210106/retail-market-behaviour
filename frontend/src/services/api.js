const API_URL = "http://127.0.0.1:8000";

async function fetchAPI(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_URL}${endpoint}`, options);
        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`Error fetching ${endpoint}:`, error);
        throw error;
    }
}

// Home
export const getHome = () => fetchAPI("/");

// Default Data
export const getDefaultData = () => fetchAPI("/default-data");

// Peak Sales Insights
export const getPeakSales = () => fetchAPI("/peak-sales");

// KNN Similar Products
export const getSimilarProducts = (product) => fetchAPI(`/similar-products?product=${encodeURIComponent(product)}`);
export const getAllSimilarProducts = () => fetchAPI("/similar-products/all");

// Apriori - Frequently Bought Together
export const getFrequentlyBoughtTogether = (minSupport = 0.001, minConfidence = 0.01, topK = 20) =>
    fetchAPI(`/frequently-bought-together?min_support=${minSupport}&min_confidence=${minConfidence}&top_k=${topK}`);

// K-Means Clustering
export const getCustomerSegmentation = (k = 3) => fetchAPI(`/customer-segmentation?k=${k}`);

// Decision Tree - Spend Prediction
export const getSpendPrediction = () => fetchAPI("/customer-spend-prediction");

// PCA Visualization
export const getPcaVisualization = () => fetchAPI("/pca-visualization");

// Customer Behavior (PCA alias)
export const getCustomerBehavior = () => fetchAPI("/customer-behavior");
