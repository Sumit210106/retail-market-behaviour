import { useEffect, useState } from "react";
import { API } from "../services/api";
import DashboardLayout from "../layouts/DashboardLayout.jsx";

export default function SimilarProductsDashboard() {
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(true);
  const [selectedProduct, setSelectedProduct] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const res = await API.allSimilarProducts();
        setData(res);
        if (Object.keys(res).length > 0) {
          setSelectedProduct(Object.keys(res)[0]);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const productList = Object.keys(data);
  const filteredProducts = productList.filter(product =>
    product.toLowerCase().includes(searchTerm.toLowerCase())
  );
  const similarItems = selectedProduct ? data[selectedProduct] : [];

  return (
    <DashboardLayout>
      <div className="p-6 bg-gray-50 min-h-screen">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-7">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              Product Similarity Search
            </h1>
            <p className="text-md text-gray-500">
              Using K-Nearest Neighbors algorithm
            </p>
          </div>

          {loading ? (
            <div className="text-gray-500 text-center py-20">Loading products...</div>
          ) : (
            <>
              {/* Stats Row */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-7">
                <div className="bg-white rounded-lg p-4 border border-gray-200 ">
                  <div className="text-2xl font-bold text-indigo-600">{productList.length}</div>
                  <div className="text-sm text-gray-600 mt-1">Total Products</div>
                </div>
                <div className="bg-white rounded-lg p-4 border border-gray-200 ">
                  <div className="text-2xl font-bold text-indigo-600">{similarItems.length}</div>
                  <div className="text-sm text-gray-600 mt-1">Similar Matches</div>
                </div>
              </div>

              <div className="grid lg:grid-cols-3 gap-6">
                {/* Left: Product List */}
                <div className="lg:col-span-1">
                  <div className="bg-white rounded-lg border border-gray-200  p-5">
                    <h2 className="font-semibold text-gray-800 mb-1">Products</h2>
                    <p className="text-sm text-gray-500 mb-4">Select one to explore</p>

                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm mb-4 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent"
                    />

                    <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
                      {filteredProducts.map((product, idx) => (
                        <button
                          key={idx}
                          onClick={() => setSelectedProduct(product)}
                          className={`w-full text-left px-3 py-2.5 rounded-md text-sm transition-colors ${
                            selectedProduct === product
                              ? "bg-indigo-100 text-indigo-900 font-medium border border-indigo-300"
                              : "text-gray-700 hover:bg-gray-50 border border-transparent"
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="truncate">{product}</span>
                            <span className="text-xs text-gray-400 ml-2">{data[product].length}</span>
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Right: Similar Products */}
                <div className="lg:col-span-2">
                  <div className="bg-white rounded-lg border border-gray-200  p-5">
                    <div className="mb-5">
                      <h2 className="font-semibold text-gray-800 mb-1">
                        Similar to: {selectedProduct ? `"${selectedProduct}"` : "Nothing selected"}
                      </h2>
                      <p className="text-sm text-gray-500">
                        {similarItems.length > 0 
                          ? `Found ${similarItems.length} similar products`
                          : "Choose a product to see recommendations"}
                      </p>
                    </div>

                    {similarItems.length === 0 ? (
                      <div className="text-center py-20 text-gray-400">
                        <p>No product selected yet</p>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {similarItems.map((item, idx) => {
                          const matchPercent = (item.similarity * 100).toFixed(0);
                          const isStrong = item.similarity >= 0.7;
                          const isMedium = item.similarity >= 0.5;
                          
                          return (
                            <div
                              key={idx}
                              className={`rounded-lg p-4 border transition-all ${
                                isStrong 
                                  ? "bg-indigo-50 border-indigo-200 " 
                                  : "bg-gray-50 border-gray-200 hover:border-indigo-200"
                              }`}
                            >
                              <div className="flex justify-between items-start mb-3">
                                <div className="flex-1 pr-3">
                                  <h3 className="font-medium text-gray-800 mb-1">
                                    {item.product}
                                  </h3>
                                  <div className="flex items-center gap-2">
                                    <span className={`text-sm font-medium ${
                                      isStrong ? "text-indigo-700" : isMedium ? "text-blue-600" : "text-gray-600"
                                    }`}>
                                      {matchPercent}% similar
                                    </span>
                                    {isStrong && (
                                      <span className="text-xs bg-indigo-200 text-indigo-800 px-2 py-0.5 rounded-full">
                                        Strong match
                                      </span>
                                    )}
                                  </div>
                                </div>
                              </div>
                              
                              <div className="w-full bg-gray-200 h-2 rounded-full overflow-hidden">
                                <div
                                  className={`h-2 rounded-full transition-all ${
                                    isStrong ? "bg-indigo-500" : isMedium ? "bg-blue-400" : "bg-gray-400"
                                  }`}
                                  style={{ width: `${matchPercent}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* API Data Preview */}
              <div className="mt-8 bg-white rounded-lg border border-gray-200  p-5">
                <h3 className="font-semibold text-gray-800 mb-1">API Response</h3>
                <p className="text-sm text-gray-500 mb-3">Raw data from the similarity engine</p>
                <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                  <pre className="text-xs text-green-400 font-mono">
                    {JSON.stringify(
                      Object.fromEntries(Object.entries(data).slice(0, 2)),
                      null,
                      2
                    )}
                  </pre>
                </div>
                <p className="text-xs text-gray-500 mt-3">
                  Showing first 2 products • {productList.length} total in database
                </p>
              </div>
            </>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}