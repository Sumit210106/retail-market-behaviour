import { useEffect, useState } from "react";
import { API } from "../services/api";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import DashboardLayout from "../layouts/DashboardLayout.jsx";

const PAGE_SIZE = 12;
const COLORS = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#4f46e5", "#7c3aed"];

export default function SpendPredictionDashboard({ sidebarWidth = 64 }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);

  useEffect(() => {
    async function load() {
      try {
        const res = await API.customerSpendPrediction();
        const sorted = res.sort((a, b) => b.predicted_spend - a.predicted_spend);
        setData(sorted);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const totalPages = Math.ceil(data.length / PAGE_SIZE);
  const paginatedData = data.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE
  );
  const maxSpend = Math.max(...data.map(item => item.predicted_spend), 1);
  const totalPredictedSpend = data.reduce((acc, cur) => acc + cur.predicted_spend, 0);
  
  // Calculate dynamic thresholds based on data distribution
  const avgSpend = totalPredictedSpend / data.length;
  const highThreshold = avgSpend * 1.5;
  const mediumThreshold = avgSpend * 0.75;

  const getPagination = () => {
    const pages = [];
    for (let i = 1; i <= totalPages; i++) {
      if (
        i <= 2 ||
        i > totalPages - 2 ||
        (i >= currentPage - 1 && i <= currentPage + 1)
      ) {
        pages.push(i);
      } else if (
        i === 3 && currentPage > 4 ||
        i === totalPages - 2 && currentPage < totalPages - 3
      ) {
        pages.push("...");
      }
    }
    return pages;
  };

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-50 p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Customer Spend Predictions
          </h1>
          <p className="text-gray-600 mb-1">
            See which customers are likely to spend more so you can focus your efforts where it matters most.
          </p>
          <p className="text-sm text-gray-500">
            Using Decision Tree analysis on past purchase behavior
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-500">Loading predictions...</div>
          </div>
        ) : (
          <>
            {/* Overview Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              {/* Pie Chart */}
              <div className="lg:col-span-2 bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <div className="mb-4">
                  <h2 className="text-lg font-semibold text-gray-800 mb-1">
                    Top 6 Customers
                  </h2>
                  <p className="text-sm text-gray-600">
                    Your biggest spenders at a glance
                  </p>
                </div>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={data.slice(0, 6)}
                        dataKey="predicted_spend"
                        nameKey="CustomerID"
                        cx="50%"
                        cy="50%"
                        outerRadius={90}
                        label={(entry) => `$${entry.predicted_spend.toFixed(0)}`}
                        labelLine={true}
                      >
                        {data.slice(0, 6).map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value) => `$${value.toFixed(2)}`}
                        contentStyle={{ 
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '6px',
                          padding: '8px 12px',
                          fontSize: '14px'
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Stats Cards */}
              <div className="space-y-4">
                <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                  <p className="text-sm text-indigo-700 mb-1">Total Expected Revenue</p>
                  <p className="text-3xl font-bold text-indigo-900">
                    ${totalPredictedSpend.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                  </p>
                  <p className="text-xs text-indigo-600 mt-1">From all customers combined</p>
                </div>

                <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                  <p className="text-sm text-indigo-700 mb-1">Total Customers</p>
                  <p className="text-3xl font-bold text-indigo-900">{data.length}</p>
                  <p className="text-xs text-indigo-600 mt-1">Active in your base</p>
                </div>

                <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                  <p className="text-sm text-indigo-700 mb-1">Average per Customer</p>
                  <p className="text-3xl font-bold text-indigo-900">
                    ${(totalPredictedSpend / data.length).toLocaleString('en-US', { maximumFractionDigits: 0 })}
                  </p>
                  <p className="text-xs text-indigo-600 mt-1">Typical spend amount</p>
                </div>
              </div>
            </div>

            {/* Customer List */}
            <div className="mb-5">
              <h2 className="text-lg font-semibold text-gray-800 mb-1">
                All Customers
              </h2>
              <p className="text-sm text-gray-600">
                Sorted by expected spending, highest first
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {paginatedData.map((item, idx) => {
                const spendLevel = item.predicted_spend >= highThreshold ? "High value" : 
                                   item.predicted_spend >= mediumThreshold ? "Medium" : "Low";
                const levelColor = item.predicted_spend >= highThreshold ? "text-indigo-700 bg-indigo-100 border-indigo-300" :
                                   item.predicted_spend >= mediumThreshold ? "text-blue-700 bg-blue-100 border-blue-300" : 
                                   "text-gray-600 bg-gray-100 border-gray-300";
                const barColor = item.predicted_spend >= highThreshold ? "bg-indigo-500" :
                                 item.predicted_spend >= mediumThreshold ? "bg-blue-400" : "bg-gray-300";

                return (
                  <div
                    key={idx}
                    className="bg-white rounded-lg p-4 border border-gray-200 hover:border-indigo-300 hover:shadow-md transition-all"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <span className="text-xs text-gray-500 uppercase tracking-wide">
                        Customer
                      </span>
                      <span className={`text-xs font-medium px-2 py-1 rounded-md border ${levelColor}`}>
                        {spendLevel}
                      </span>
                    </div>
                    
                    <p className="font-semibold text-base text-gray-900 mb-3">{item.CustomerID}</p>
                    
                    <div className="mb-3">
                      <p className="text-xs text-gray-500 mb-1">
                        Expected to spend
                      </p>
                      <p className="font-bold text-xl text-gray-900">
                        ${item.predicted_spend.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                      </p>
                    </div>
                    
                    <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden">
                      <div
                        className={`h-1.5 rounded-full ${barColor}`}
                        style={{ width: `${(item.predicted_spend / maxSpend) * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Pagination */}
            <div className="flex justify-center items-center mt-8 gap-2">
              <button
                onClick={() => setCurrentPage((p) => Math.max(p - 1, 1))}
                disabled={currentPage === 1}
                className="px-4 py-2 rounded-md bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-medium text-gray-700"
              >
                Previous
              </button>
              
              <div className="flex gap-2">
                {getPagination().map((p, idx) =>
                  p === "..." ? (
                    <span key={idx} className="px-2 py-2 text-gray-400">
                      ...
                    </span>
                  ) : (
                    <button
                      key={idx}
                      onClick={() => setCurrentPage(p)}
                      className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        currentPage === p
                          ? "bg-indigo-600 text-white"
                          : "bg-white border border-gray-300 text-gray-700 hover:bg-gray-50"
                      }`}
                    >
                      {p}
                    </button>
                  )
                )}
              </div>

              <button
                onClick={() => setCurrentPage((p) => Math.min(p + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="px-4 py-2 rounded-md bg-white border border-gray-300 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-medium text-gray-700"
              >
                Next
              </button>
            </div>

            <div className="text-center mt-3 text-sm text-gray-500">
              Page {currentPage} of {totalPages}
            </div>

            {/* API Response Preview */}
            <div className="mt-10 bg-white rounded-lg p-6 border border-gray-200">
              <div className="mb-3">
                <h2 className="text-lg font-semibold text-gray-800 mb-1">
                  API Response Data
                </h2>
                <p className="text-sm text-gray-600">
                  Raw JSON data from the prediction model
                </p>
              </div>
              <div className=" rounded-md border border-black/20 bg-black p-4 overflow-x-auto">
                <pre className="text-sm text-green-500 font-mono">
                  {JSON.stringify(data.slice(0, 5), null, 2)}
                </pre>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Showing first 5 records • Total: {data.length} customers
              </p>
            </div>
          </>
        )}
      </div>
    </DashboardLayout>
  );
}