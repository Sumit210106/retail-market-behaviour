import { useEffect, useState } from "react";
import { API } from "../services/api";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import DashboardLayout from "../layouts/DashboardLayout.jsx";

export default function BehaviorMap() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await API.customerBehavior();
        setData(res);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Indigo theme colors for scatter points
  const COLORS = ["#6366f1", "#8b5cf6", "#ec4899", "#10b981", "#f59e0b"];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const dataPoint = payload[0].payload;
      return (
        <div className="bg-white border border-indigo-100 p-3 rounded-lg shadow-lg">
          <p className="font-bold text-indigo-900 mb-1">Customer #{dataPoint.customer_id}</p>
          <div className="space-y-1 text-xs text-gray-600">
            <p>Total Spend: <span className="font-semibold text-gray-900">${dataPoint.total_spend}</span></p>
            <p>Orders: <span className="font-semibold text-gray-900">{dataPoint.total_orders}</span></p>
            <p>Items: <span className="font-semibold text-gray-900">{dataPoint.total_items}</span></p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <DashboardLayout>
      <div className="min-h-screen bg-gray-50 p-6">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Customer Behavior Map
          </h1>
          <p className="text-gray-600 mb-1">
            Visualizing customer similarities using PCA (Principal Component Analysis).
          </p>
          <p className="text-sm text-gray-500">
            Customers closer together exhibit similar purchasing behaviors
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="text-gray-500">Loading visualization...</div>
          </div>
        ) : (
          <>
            {/* Stats Overview */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                <p className="text-sm text-indigo-700 mb-1">Explained Variance (PC1)</p>
                <p className="text-3xl font-bold text-indigo-900">
                  {(data?.explained_variance[0] * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-indigo-600 mt-1">Primary behavior factor</p>
              </div>
              <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                <p className="text-sm text-indigo-700 mb-1">Explained Variance (PC2)</p>
                <p className="text-3xl font-bold text-indigo-900">
                  {(data?.explained_variance[1] * 100).toFixed(1)}%
                </p>
                <p className="text-xs text-indigo-600 mt-1">Secondary behavior factor</p>
              </div>
              <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
                <p className="text-sm text-indigo-700 mb-1">Total Customers</p>
                <p className="text-3xl font-bold text-indigo-900">
                  {data?.data.length}
                </p>
                <p className="text-xs text-indigo-600 mt-1">Analyzed in 2D space</p>
              </div>
            </div>

            {/* Scatter Chart */}
            <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200 mb-8">
              <div className="mb-6">
                <h2 className="text-lg font-semibold text-gray-800">
                  Behavioral Clusters
                </h2>
                <p className="text-sm text-gray-600">
                  2D projection of multidimensional purchase data
                </p>
              </div>

              <div className="h-[500px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name="PC1"
                      tick={{ fill: '#6b7280', fontSize: 12 }}
                      tickLine={false}
                      axisLine={{ stroke: '#e5e7eb' }}
                      label={{ value: 'Principal Component 1', position: 'bottom', offset: 0, fill: '#6b7280', fontSize: 12 }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="PC2"
                      tick={{ fill: '#6b7280', fontSize: 12 }}
                      tickLine={false}
                      axisLine={{ stroke: '#e5e7eb' }}
                      label={{ value: 'Principal Component 2', angle: -90, position: 'left', fill: '#6b7280', fontSize: 12 }}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Customers" data={data?.data} fill="#6366f1">
                      {data?.data.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* API Response Preview */}
            <div className="bg-white rounded-lg p-6 border border-gray-200">
              <div className="mb-3">
                <h2 className="text-lg font-semibold text-gray-800 mb-1">
                  API Response Data
                </h2>
                <p className="text-sm text-gray-600">
                  Raw JSON output from PCA analysis
                </p>
              </div>
              <div className="rounded-md border border-black/20 bg-black p-4 overflow-x-auto max-h-96">
                <pre className="text-sm text-green-500 font-mono">
                  {JSON.stringify(data, null, 2)}
                </pre>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                Showing explained variance and projected coordinates
              </p>
            </div>
          </>
        )}
      </div>
    </DashboardLayout>
  );
}
