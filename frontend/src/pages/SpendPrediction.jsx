import { useEffect, useState } from "react";
import { API } from "../services/api";

export default function SpendPrediction() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await API.customerSpendPrediction();
        setData(res);
      } catch (err) {
        console.error("Error fetching prediction:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold mb-4">Customer Spend Prediction</h1>

      {loading && (
        <div className="text-gray-500 animate-pulse">Loading predictions...</div>
      )}

      {!loading && (
        <div className="overflow-auto border rounded-lg shadow-sm">
          <table className="min-w-full text-left">
            <thead className="bg-gray-100 border-b">
              <tr>
                <th className="px-4 py-2 font-medium">Customer ID</th>
                <th className="px-4 py-2 font-medium">Predicted Spend (£)</th>
              </tr>
            </thead>

            <tbody>
              {data.map((item, index) => (
                <tr
                  key={index}
                  className="border-b hover:bg-gray-50 transition"
                >
                  <td className="px-4 py-2">{item.CustomerID}</td>
                  <td
                    className={`px-4 py-2 font-medium ${
                      item.predicted_spend > 50
                        ? "text-green-600"
                        : item.predicted_spend === 0
                        ? "text-gray-400"
                        : "text-blue-500"
                    }`}
                  >
                    £{item.predicted_spend.toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
