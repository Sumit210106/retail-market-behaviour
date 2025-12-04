import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// Pages
import FrequentlyBought from "./pages/FrequentlyBought.jsx";
import CustomerSegments from "./pages/CustomerSegments.jsx";
import SpendPrediction from "./pages/SpendPrediction.jsx";
import OutlierDetection from "./pages/OutlierDetection.jsx";
import Recommendations from "./pages/Recommendations.jsx";
import BehaviorMap from "./pages/BehaviorMap.jsx";
import PeakInsights from "./pages/PeakInsights.jsx";
import UploadStoreData from "./pages/UploadStoreData.jsx";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<FrequentlyBought />} />
        <Route path="/frequently-bought" element={<FrequentlyBought />} />
        <Route path="/segments" element={<CustomerSegments />} />
        <Route path="/spend-prediction" element={<SpendPrediction />} />
        <Route path="/outliers" element={<OutlierDetection />} />
        <Route path="/recommendations" element={<Recommendations />} />
        <Route path="/behavior-map" element={<BehaviorMap />} />
        <Route path="/peak-insights" element={<PeakInsights />} />
        <Route path="/upload" element={<UploadStoreData />} />
      </Routes>
    </Router>
  );
}
