import { Link } from "react-router-dom";

export default function Sidebar() {
  return (
    <aside className="w-56 h-full bg-gray-900 text-white flex flex-col p-4 gap-4">
      <h2 className="text-lg font-semibold">Dashboard</h2>

      <nav className="flex flex-col gap-2 text-sm">
        <Link className="hover:bg-gray-700 p-2 rounded" to="/frequently-bought">Frequently Bought</Link>
        <Link className="hover:bg-gray-700 p-2 rounded" to="/segments">Customer Segments</Link>
        <Link className="hover:bg-gray-700 p-2 rounded" to="/spend-prediction">Spend Prediction</Link>
        <Link className="hover:bg-gray-700 p-2 rounded" to="/outliers">Outlier Detection</Link>
        <Link className="hover:bg-gray-700 p-2 rounded" to="/recommendations">Recommendations</Link>
        <Link className="hover:bg-gray-700 p-2 rounded" to="/behavior-map">Behavior Map</Link>
        <Link className="hover:bg-gray-700 p-2 rounded" to="/peak-insights">Peak Insights</Link>
        <Link className="hover:bg-gray-700 p-2 rounded" to="/upload">Upload Data</Link>
      </nav>
    </aside>
  );
}
