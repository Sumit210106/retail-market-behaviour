import { useEffect, useState } from "react";
import { API } from "../services/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line
} from "recharts";
import DashboardLayout from "../layouts/DashboardLayout.jsx";

export default function PeakSalesDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await API.peakSales();
        setData(res);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) {
    return (
      <DashboardLayout>
        <div className="p-6 bg-gray-50 min-h-screen">
          <div className="text-gray-500 text-center py-20">Loading sales data...</div>
        </div>
      </DashboardLayout>
    );
  }

  const bestHour = data?.best_hour || "N/A";
  const bestDay = data?.best_day || "N/A";
  
  // Convert hourly sales object to array
  const hourlyData = data?.hourly_sales 
    ? Object.entries(data.hourly_sales).map(([hour, sales]) => ({
        hour: parseInt(hour),
        sales: sales
      })).sort((a, b) => a.hour - b.hour)
    : [];

  // Convert weekday sales object to array with proper ordering
  const dayOrder = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const weekdayData = data?.weekday_sales
    ? Object.entries(data.weekday_sales).map(([day, sales]) => ({
        day: day,
        sales: sales
      })).sort((a, b) => dayOrder.indexOf(a.day) - dayOrder.indexOf(b.day))
    : [];

  const totalWeeklySales = weekdayData.reduce((acc, item) => acc + item.sales, 0);
  const totalHourlySales = hourlyData.reduce((acc, item) => acc + item.sales, 0);
  const avgDailySales = weekdayData.length > 0 ? totalWeeklySales / weekdayData.length : 0;

  return (
    <DashboardLayout>
      <div className="p-6 bg-gray-50 min-h-screen">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="mb-7">
            <h1 className="text-2xl font-semibold text-gray-800 mb-2">
              Peak Sales Analysis
            </h1>
            <p className="text-gray-600">
              Find out when your business is busiest and plan accordingly
            </p>
            <p className="text-gray-500 text-sm mt-1">
              Based on historical transaction patterns
            </p>
          </div>

          {/* Key Insights */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-7">
            <div className="bg-white rounded-lg p-5 border border-gray-200 shadow-sm">
              <div className="text-sm text-gray-600 mb-1">Best Day</div>
              <div className="text-2xl font-bold text-indigo-600">{bestDay}</div>
              <div className="text-xs text-gray-500 mt-1">Highest sales day</div>
            </div>

            <div className="bg-white rounded-lg p-5 border border-gray-200 shadow-sm">
              <div className="text-sm text-gray-600 mb-1">Best Hour</div>
              <div className="text-2xl font-bold text-indigo-600">{bestHour}:00</div>
              <div className="text-xs text-gray-500 mt-1">Peak selling time</div>
            </div>

            <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
              <div className="text-sm text-indigo-700 mb-1">Weekly Total</div>
              <div className="text-2xl font-bold text-indigo-900">
                ${totalWeeklySales.toLocaleString()}
              </div>
              <div className="text-xs text-indigo-600 mt-1">All days combined</div>
            </div>

            <div className="bg-indigo-50 rounded-lg p-5 border border-indigo-200">
              <div className="text-sm text-indigo-700 mb-1">Daily Average</div>
              <div className="text-2xl font-bold text-indigo-900">
                ${avgDailySales.toLocaleString('en-US', { maximumFractionDigits: 0 })}
              </div>
              <div className="text-xs text-indigo-600 mt-1">Per day of week</div>
            </div>
          </div>

          {/* Charts */}
          <div className="space-y-6">
            {/* Day of Week Sales */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6">
              <div className="mb-5">
                <h2 className="font-semibold text-gray-800 mb-1">Sales by Day of Week</h2>
                <p className="text-sm text-gray-500">Which days bring in the most revenue</p>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={weekdayData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis 
                      dataKey="day" 
                      tick={{ fill: '#6b7280', fontSize: 12 }}
                    />
                    <YAxis 
                      tick={{ fill: '#6b7280', fontSize: 12 }}
                      label={{ value: 'Sales ($)', angle: -90, position: 'insideLeft', style: { fill: '#6b7280' } }}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'white',
                        border: '1px solid #e5e7eb',
                        borderRadius: '6px',
                        padding: '8px 12px'
                      }}
                      formatter={(value) => [`$${value.toLocaleString()}`, 'Sales']}
                    />
                    <Bar 
                      dataKey="sales" 
                      fill="#6366f1" 
                      radius={[6, 6, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Hourly Sales */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm p-6">
              <div className="mb-5">
                <h2 className="font-semibold text-gray-800 mb-1">Sales Throughout the Day</h2>
                <p className="text-sm text-gray-500">Hourly breakdown to spot your rush hours</p>
              </div>
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={hourlyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis 
                      dataKey="hour" 
                      tick={{ fill: '#6b7280', fontSize: 12 }}
                      label={{ value: 'Hour (24h format)', position: 'insideBottom', offset: -5, style: { fill: '#6b7280' } }}
                    />
                    <YAxis 
                      tick={{ fill: '#6b7280', fontSize: 12 }}
                      label={{ value: 'Sales ($)', angle: -90, position: 'insideLeft', style: { fill: '#6b7280' } }}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'white',
                        border: '1px solid #e5e7eb',
                        borderRadius: '6px',
                        padding: '8px 12px'
                      }}
                      formatter={(value) => [`$${value.toLocaleString()}`, 'Sales']}
                      labelFormatter={(hour) => `${hour}:00`}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="sales" 
                      stroke="#8b5cf6" 
                      strokeWidth={3}
                      dot={{ fill: '#8b5cf6', r: 4 }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Quick Insights */}
            <div className="grid md:grid-cols-2 gap-4">
              <div className="bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg p-5 border border-indigo-200">
                <h3 className="font-semibold text-gray-800 mb-2">💡 Business Insight</h3>
                <p className="text-sm text-gray-700 leading-relaxed">
                  Your busiest time is <strong>{bestDay}s at {bestHour}:00</strong>. 
                  Consider scheduling more staff during this peak period to handle the rush.
                </p>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-5 border border-blue-200">
                <h3 className="font-semibold text-gray-800 mb-2">📊 Pattern Analysis</h3>
                <p className="text-sm text-gray-700 leading-relaxed">
                  Daily average is <strong>${avgDailySales.toFixed(0)}</strong>. 
                  Days above this average might be good for running promotions or special events.
                </p>
              </div>
            </div>
          </div>

          {/* API Data Preview */}
          <div className="mt-8 bg-white rounded-lg border border-gray-200 shadow-sm p-5">
            <h3 className="font-semibold text-gray-800 mb-1">API Response</h3>
            <p className="text-sm text-gray-500 mb-3">Raw peak sales data from the server</p>
            <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
              <pre className="text-xs text-green-400 font-mono">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}