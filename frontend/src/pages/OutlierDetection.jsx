import DashboardLayout from "../layouts/DashboardLayout.jsx";

export default function OutlierDetection() {
  return (
    <DashboardLayout>
      <h1 className="text-2xl font-semibold mb-4">Outlier Detection</h1>
      <p>DBSCAN anomalies will be displayed here.</p>
    </DashboardLayout>
  );
}
