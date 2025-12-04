export default function FileUploader({ onUpload }) {
  return (
    <div className="border-dashed border-2 border-gray-300 p-6 rounded-md text-center">
      <input
        type="file"
        accept=".csv,.xlsx"
        className="cursor-pointer"
        onChange={(e) => onUpload(e.target.files[0])}
      />
      <p className="text-gray-500 mt-2 text-sm">Upload CSV or Excel</p>
    </div>
  );
}
