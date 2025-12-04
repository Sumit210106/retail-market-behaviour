import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  ShoppingBag,
  Users,
  TrendingUp,
  AlertTriangle,
  ThumbsUp,
  Map,
  BarChart2,
  UploadCloud,
  X
} from "lucide-react";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

const menuItems = [
  // { icon: ShoppingBag, label: "Frequently Bought", path: "/frequently-bought" },
  { icon: Users, label: "Customer Segments", path: "/segments" },
  { icon: TrendingUp, label: "Spend Prediction", path: "/spend-prediction" },
  // { icon: AlertTriangle, label: "Outlier Detection", path: "/outliers" },
  { icon: ThumbsUp, label: "Recommendations", path: "/recommendations" },
  { icon: Map, label: "Behavior Map", path: "/behavior-map" },
  { icon: BarChart2, label: "Peak Insights", path: "/peak-insights" },
  { icon: UploadCloud, label: "Upload Data", path: "/upload" },
];

export default function Sidebar({ isOpen, onClose }) {
  const location = useLocation();

  return (
    <>
      {/* Mobile Overlay */}
      <div
        className={cn(
          "fixed inset-0 bg-black/50 z-40 lg:hidden transition-opacity duration-300",
          isOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        )}
        onClick={onClose}
      />

      {/* Sidebar Container */}
      <aside
        className={cn(
          "fixed lg:static inset-y-0 left-0 z-50 w-64 bg-white border-r border-gray-200 transform transition-transform duration-300 ease-in-out lg:transform-none lg:translate-x-0 flex flex-col",
          isOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        {/* Header */}
        <div className="h-16 flex items-center justify-between px-6 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <LayoutDashboard className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">MinersAI</span>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden p-1 hover:bg-gray-100 rounded-md"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          {menuItems.map((item) => {
            const isActive = location.pathname === item.path;
            const Icon = item.icon;

            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => onClose && onClose()} 
                className={cn(
                  "flex items-center gap-3 px-5 py-4 rounded-lg text-base font-medium transition-colors",
                  isActive
                    ? "bg-indigo-50 text-indigo-700"
                    : "text-gray-600 hover:bg-gray-50 hover:text-gray-900"
                )}
              >
                <Icon className={cn("w-5 h-5", isActive ? "text-indigo-600" : "text-gray-400 group-hover:text-gray-500")} />
                {item.label}
              </Link>
            );
          })}
        </nav>

      </aside>
    </>
  );
}
