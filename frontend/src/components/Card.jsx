export default function Card({ title, children }) {
  return (
    <div className="bg-white rounded-lg shadow p-4 border">
      <h3 className="text-md font-semibold mb-2">{title}</h3>
      {children}
    </div>
  );
}
