import { useEffect, useState } from "react";
import { API } from "../services/api";
import DashboardLayout from "../layouts/DashboardLayout.jsx";

export default function AprioriDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const [searchTerm, setSearchTerm] = useState("");
  const [selectedItemset, setSelectedItemset] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const res = await API.apriori();
        setData(res);

        // Default select the top-support itemset
        if (res?.frequent_itemsets?.length > 0) {
          const sorted = [...res.frequent_itemsets].sort(
            (a, b) => b.support - a.support
          );
          setSelectedItemset(sorted[0]);
        }
      } catch (err) {
        console.error("Apriori API Error:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading)
    return (
      <DashboardLayout>
        <div className="min-h-screen flex items-center justify-center text-gray-500">
          Loading Apriori Market Basket Analysis…
        </div>
      </DashboardLayout>
    );

  // Raw API data
  const itemsets = data?.frequent_itemsets || [];
  const rules = data?.rules || [];

  // ---- SORTING LOGIC (FRONTEND) ----
  const sortedItemsets = [...itemsets].sort((a, b) => b.support - a.support);
  const sortedRules = [...rules].sort((a, b) => b.lift - a.lift);
  // ----------------------------------

  // Filter itemsets by search
  const filteredItemsets = sortedItemsets.filter((i) =>
    i.itemsets.join(", ").toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Rules where the selected itemset is the antecedent
  const selectedRules = selectedItemset
    ? sortedRules.filter(
        (r) => {
          const a = [...r.antecedents].sort().join("|");
          const b = [...selectedItemset.itemsets].sort().join("|");
          return a === b;
        }
      )
    : [];

  return (
    <DashboardLayout>
      <div className="p-6 bg-gray-50 min-h-screen">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-7">
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              Market Basket — Apriori / FP-Growth
            </h1>
            <p className="text-md text-gray-500">
              Frequent itemsets + association rules powered by FP-Growth
            </p>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-7">
            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="text-2xl font-bold text-indigo-600">
                {sortedItemsets.length}
              </div>
              <div className="text-sm text-gray-600 mt-1">
                Frequent Itemsets
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="text-2xl font-bold text-indigo-600">
                {sortedRules.length}
              </div>
              <div className="text-sm text-gray-600 mt-1">
                Association Rules
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="text-2xl font-bold text-indigo-600">
                {data.meta?.max_itemset_len || "N/A"}
              </div>
              <div className="text-sm text-gray-600 mt-1">
                Max Itemset Length
              </div>
            </div>

            <div className="bg-white rounded-lg p-4 border border-gray-200">
              <div className="text-2xl font-bold text-indigo-600">
                {data.meta?.min_support || "N/A"}
              </div>
              <div className="text-sm text-gray-600 mt-1">Min Support</div>
            </div>
          </div>

          {/* Content */}
          <div className="grid lg:grid-cols-3 gap-6">
            {/* LEFT: Frequent Itemsets */}
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg border border-gray-200 p-5">
                <h2 className="font-semibold text-gray-800 mb-1">
                  Frequent Itemsets
                </h2>
                <p className="text-sm text-gray-500 mb-4">
                  Select an itemset to see its rules
                </p>

                <input
                  type="text"
                  placeholder="Search itemset..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm mb-4 focus:outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent"
                />

                <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
                  {filteredItemsets.map((i, index) => {
                    const label = i.itemsets.join(", ");
                    const support = (i.support * 100).toFixed(2);

                    return (
                      <button
                        key={index}
                        onClick={() => setSelectedItemset(i)}
                        className={`w-full text-left px-3 py-2 rounded-md text-sm transition-colors ${
                          selectedItemset === i
                            ? "bg-indigo-100 text-indigo-900 font-medium border border-indigo-300"
                            : "text-gray-700 hover:bg-gray-50 border border-transparent"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="truncate">{label}</span>
                          <span className="text-xs text-gray-400 ml-2">
                            {support}%
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* RIGHT: Rules */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg border border-gray-200 p-5">
                <h2 className="font-semibold text-gray-800 mb-1">
                  Rules for:{" "}
                  {selectedItemset ? selectedItemset.itemsets.join(", ") : "—"}
                </h2>
                <p className="text-sm text-gray-500 mb-5">
                  {selectedRules.length} rules found
                </p>

                {selectedRules.length === 0 ? (
                  <div className="text-center py-20 text-gray-400">
                    <p>No rules available for this itemset</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {selectedRules.map((rule, idx) => {
                      const lift = rule.lift.toFixed(2);
                      const confidence = (rule.confidence * 100).toFixed(1);

                      return (
                        <div
                          key={idx}
                          className="rounded-lg p-4 border bg-gray-50 hover:border-indigo-300 transition-all"
                        >
                          <div className="mb-2">
                            <h3 className="font-medium text-gray-800">
                              {rule.antecedents.join(", ")} ➜{" "}
                              {rule.consequents.join(", ")}
                            </h3>
                          </div>

                          <div className="text-sm text-gray-600 mb-3">
                            Lift: {lift} • Confidence: {confidence}%
                          </div>

                          <div className="w-full bg-gray-200 h-2 rounded-full overflow-hidden">
                            <div
                              className="h-2 bg-indigo-500 rounded-full"
                              style={{ width: `${confidence}%` }}
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

          {/* API Preview */}
          <div className="mt-8 bg-white rounded-lg border border-gray-200 p-5">
            <h3 className="font-semibold text-gray-800 mb-1">
              API Response Preview
            </h3>
            <p className="text-sm text-gray-500 mb-3">
              Showing first 2 sorted itemsets + first 2 sorted rules
            </p>

            <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-xs text-green-400 font-mono">
                {JSON.stringify(
                  {
                    itemsets: sortedItemsets.slice(0, 2),
                    rules: sortedRules.slice(0, 2),
                  },
                  null,
                  2
                )}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
