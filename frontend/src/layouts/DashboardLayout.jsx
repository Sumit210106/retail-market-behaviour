import Sidebar from "../components/Sidebar.jsx";
import Navbar from "../components/Navbar.jsx";

export default function DashboardLayout({ children }) {
  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />

      <div className="flex-1 flex flex-col">
        <Navbar />
        <main className="p-6 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}
