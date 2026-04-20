import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import DashboardLayout from "../layouts/DashboardLayout.jsx";

const COLORS = ["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#4f46e5", "#7c3aed"];

import { API } from "../services/api";

export default function CustomerSegments() {
  const [segments, setSegments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    API.customerSegmentation(3)
      .then((data) => {
        setSegments(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const totalCustomers = segments.reduce((sum, s) => sum + s.customers, 0);
  const maxSpend = Math.max(...segments.map(s => parseFloat(s.avg_spend)), 1);

  // Mock trend data for visualization
  const trendData = [
    { month: 'Jan', c0: 65, c1: 2800, c2: 1600 },
    { month: 'Feb', c0: 58, c1: 2950, c2: 1750 },
    { month: 'Mar', c0: 72, c1: 3100, c2: 1680 },
    { month: 'Apr', c0: 68, c1: 2850, c2: 1850 },
    { month: 'May', c0: 85, c1: 3300, c2: 1950 },
    { month: 'Jun', c0: 78, c1: 3500, c2: 2100 },
  ];

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-50 p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Customer Segmentation
          </h1>
          <p className="text-gray-600 mb-1">
            Group customers based on purchasing behavior using K-Means.
          </p>
          <p className="text-sm text-gray-500">
            Identify behavioral archetypes for targeted marketing
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-500">Loading segments...</div>
          </div>
        ) : (
          <>
            {/* Overview Section */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              {/* Trend Chart */}
              <div className="lg:col-span-2 bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <div className="mb-4">
                  <h2 className="text-lg font-semibold text-gray-800 mb-1">
                    Spending Trends
                  </h2>
                  <p className="text-sm text-gray-600">
                    Average spending behavior over the last 6 months
                  </p>
                </div>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trendData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                      <XAxis
                        dataKey="month"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#6b7280', fontSize: 12 }}
                        dy={10}
                      />
                      <YAxis
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#6b7280', fontSize: 12 }}
                        tickFormatter={(value) => `$${value}`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: 'white',
                          border: '1px solid #e5e7eb',
                          borderRadius: '6px',
                          padding: '8px 12px',
                          fontSize: '14px'
                        }}
                        formatter={(value) => [`$${value}`, undefined]}
                      />
                      <Legend wrapperStyle={{ paddingTop: '20px' }} />
                      <Line type="monotone" dataKey="c0" name="Cluster 0" stroke={COLORS[0]} strokeWidth={3} dot={{ r: 4, fill: COLORS[0], strokeWidth: 2, stroke: '#fff' }} activeDot={{ r: 6 }} />
                      <Line type="monotone" dataKey="c1" name="Cluster 1" stroke={COLORS[1]} strokeWidth={3} dot={{ r: 4, fill: COLORS[1], strokeWidth: 2, stroke: '#fff' }} activeDot={{ r: 6 }} />
                      <Line type="monotone" dataKey="c2" name="Cluster 2" stroke={COLORS[2]} strokeWidth={3} dot={{ r: 4, fill: COLORS[2], strokeWidth: 2, stroke: '#fff' }} activeDot={{ r: 6 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Stats Cards */}
              <div className="space-y-4">
                <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                  <p className="text-sm text-indigo-700 mb-1">Total Customers</p>
                  <p className="text-3xl font-bold text-indigo-900">
                    {totalCustomers.toLocaleString()}
                  </p>
                  <p className="text-xs text-indigo-600 mt-1">Across all segments</p>
                </div>

                <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                  <p className="text-sm text-indigo-700 mb-1">Active Clusters</p>
                  <p className="text-3xl font-bold text-indigo-900">{segments.length}</p>
                  <p className="text-xs text-indigo-600 mt-1">Distinct behavioral groups</p>
                </div>

                <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                  <p className="text-sm text-indigo-700 mb-1">Highest Avg Spend</p>
                  <p className="text-3xl font-bold text-indigo-900">
                    ${maxSpend.toLocaleString('en-US', { maximumFractionDigits: 0 })}
                  </p>
                  <p className="text-xs text-indigo-600 mt-1">Per customer in top cluster</p>
                </div>
              </div>
            </div>

            {/* Cluster List */}
            <div className="mb-5">
              <h2 className="text-lg font-semibold text-gray-800 mb-1">
                Cluster Details
              </h2>
              <p className="text-sm text-gray-600">
                Detailed metrics for each customer segment
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {segments.map((seg, idx) => {
                const spendRatio = parseFloat(seg.avg_spend) / maxSpend;
                const barColor = spendRatio > 0.6 ? "bg-indigo-500" : spendRatio > 0.3 ? "bg-blue-400" : "bg-gray-300";

                return (
                  <div
                    key={seg.cluster_id}
                    className="bg-white rounded-lg p-4 border border-gray-200 hover:border-indigo-300 hover:shadow-md transition-all"
                  >
                    <div className="flex justify-between items-start mb-3">
                      <span className="text-xs text-gray-500 uppercase tracking-wide">
                        Segment
                      </span>
                      <span className="text-xs font-medium px-2 py-1 rounded-md border text-indigo-700 bg-indigo-100 border-indigo-300">
                        Cluster {seg.cluster_id}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <p className="text-xs text-gray-500 mb-0.5">Avg Spend</p>
                        <p className="font-bold text-lg text-gray-900">${parseFloat(seg.avg_spend).toFixed(2)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-0.5">Customers</p>
                        <p className="font-bold text-lg text-gray-900">{seg.customers}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-0.5">Avg Orders</p>
                        <p className="font-bold text-lg text-gray-900">{parseFloat(seg.avg_orders).toFixed(1)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-500 mb-0.5">Avg Items</p>
                        <p className="font-bold text-lg text-gray-900">{parseFloat(seg.avg_items).toFixed(1)}</p>
                      </div>
                    </div>

                    <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden">
                      <div
                        className={`h-1.5 rounded-full ${barColor}`}
                        style={{ width: `${spendRatio * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* API Response Preview */}
            <div className="mt-10 bg-white rounded-lg p-6 border border-gray-200">
              <div className="mb-3">
                <h2 className="text-lg font-semibold text-gray-800 mb-1">
                  API Response Data
                </h2>
                <p className="text-sm text-gray-600">
                  Raw JSON data from the segmentation model
                </p>
              </div>
              <div className="rounded-md border border-black/20 bg-black p-4 overflow-x-auto">
                <pre className="text-sm text-green-500 font-mono">
                  {JSON.stringify(segments, null, 2)}
                </pre>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Total: {segments.length} clusters
              </p>
            </div>
          </>
        )}
      </div>
    </DashboardLayout>
  );
}
