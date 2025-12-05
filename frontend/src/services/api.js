// src/services/api.js

const BASE_URL = "https://retail-market-behaviour.onrender.com"; 
// ↑ Replace with your actual backend URL

// Generic GET helper
async function get(endpoint) {
  try {
    const res = await fetch(`${BASE_URL}${endpoint}`);
    if (!res.ok) throw new Error(`API Error: ${res.status}`);
    return await res.json();
  } catch (err) {
    console.error("Fetch Error:", err);
    throw err;
  }
}

export const API = {
  // Home / health check
  home: () => get("/"),

  // Default dataset info
  defaultData: () => get("/default-data"),

  // Peak sales insights
  peakSales: () => get("/peak-sales"),

  // KNN - Similar products
  similarProducts: (product) => get(`/similar-products?product=${product}`),

  allSimilarProducts: () => get("/similar-products/all"),

  // K-Means customer segmentation
  customerSegmentation: (k = 3) => get(`/customer-segmentation?k=${k}`),

  // Decision Tree
  customerSpendPrediction: () => get("/customer-spend-prediction"),

  // PCA visualizations
  // pcaVisualization: () => get("/pca-visualization"),

  // PCA duplicate route
  customerBehavior: () => get("/customer-behavior"),

  // APRIORI ENDPOINT
  apriori: () => get("/apriori"),
};


/*
------------------------------------------------------------
HOW TO USE INSIDE FRONTEND (React Example)
------------------------------------------------------------

import { API } from "../services/api";
import { useEffect, useState } from "react";

function DemoComponent() {
  const [data, setData] = useState(null);

  useEffect(() => {
    async function loadData() {
      const res = await API.defaultData();   // <-- Call API here
      setData(res);
    }
    loadData();
  }, []);

  return (
    <div>
      <h1>Default Data Preview</h1>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </div>
  );
}

export default DemoComponent;

------------------------------------------------------------
OTHER API USAGE EXAMPLES
------------------------------------------------------------

// Peak Sales
const peak = await API.peakSales();

// Similar products
const rec = await API.similarProducts("Laptop");

// All similar products (first 50 only)
const all = await API.allSimilarProducts();

// K-Means Segmentation
const clusters = await API.customerSegmentation(3);

// Decision Tree results
const tree = await API.customerSpendPrediction();

// PCA Visualization
const pca = await API.pcaVisualization();

------------------------------------------------------------
*/
