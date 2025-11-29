import DashboardLayout from "../layouts/DashboardLayout.jsx";
import FileUploader from "../components/FileUploader.jsx";

export default function UploadStoreData() {
  const handleUpload = (file) => {
    console.log("Uploaded file:", file);
  };

  return (
    <DashboardLayout>
      <h1 className="text-2xl font-semibold mb-4">Upload Store Data</h1>
      <FileUploader onUpload={handleUpload} />
    </DashboardLayout>
  );
}
