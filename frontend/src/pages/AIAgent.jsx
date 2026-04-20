import { useState, useRef, useEffect } from "react";
import { API } from "../services/api";
import DashboardLayout from "../layouts/DashboardLayout.jsx";
import { Send, Bot, User, Clock, AlertTriangle } from "lucide-react";

export default function AIAgent() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi! I'm your AI Business Analyst. Ask me anything about your retail data.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput("");

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage },
    ]);

    setIsLoading(true);

    try {
      const res = await API.aiAgent(userMessage);

      if (res.status === "error") {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: `Error: ${res.message}`,
            isError: true,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: res.response,
            time_taken: res.time_taken,
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Unable to connect to server. Please check if backend is running.",
          isError: true,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="flex flex-col h-[calc(100vh-4rem)] bg-gray-50">

        {/* HEADER */}
        <header className="flex items-center gap-4 px-6 py-4 border-b border-gray-200 bg-white">
          <div className="w-10 h-10 bg-indigo-600 rounded-lg flex items-center justify-center">
            <Bot className="text-white w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">
              AI Business Analyst
            </h1>
            <p className="text-sm text-gray-500">
              Ask questions about your data, models, or trends
            </p>
          </div>
        </header>

        {/* CHAT AREA */}
        <div className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
          <div className="max-w-3xl mx-auto space-y-6">

            {messages.map((msg, idx) => {
              const isUser = msg.role === "user";

              return (
                <div
                  key={idx}
                  className={`flex gap-3 ${
                    isUser ? "justify-end" : "justify-start"
                  }`}
                >
                  {/* Avatar */}
                  <div
                    className={`w-9 h-9 rounded-full flex items-center justify-center ${
                      isUser
                        ? "bg-indigo-600 text-white"
                        : msg.isError
                        ? "bg-red-100 text-red-500"
                        : "bg-gray-200 text-gray-700"
                    }`}
                  >
                    {isUser ? (
                      <User className="w-4 h-4" />
                    ) : msg.isError ? (
                      <AlertTriangle className="w-4 h-4" />
                    ) : (
                      <Bot className="w-4 h-4" />
                    )}
                  </div>

                  {/* Message */}
                  <div
                    className={`max-w-[75%] px-4 py-3 rounded-lg text-sm leading-relaxed ${
                      isUser
                        ? "bg-indigo-600 text-white rounded-br-none"
                        : msg.isError
                        ? "bg-red-50 text-red-600 border border-red-200"
                        : "bg-white border border-gray-200 text-gray-800 rounded-bl-none"
                    }`}
                  >
                    {msg.content}

                    {!isUser && msg.time_taken && (
                      <div className="flex items-center gap-1 text-xs text-gray-400 mt-2">
                        <Clock className="w-3 h-3" />
                        {msg.time_taken}s
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* Loading */}
            {isLoading && (
              <div className="flex gap-3">
                <div className="w-9 h-9 rounded-full bg-gray-200 flex items-center justify-center">
                  <Bot className="w-4 h-4 text-gray-600" />
                </div>
                <div className="bg-white border border-gray-200 px-4 py-3 rounded-lg text-sm text-gray-500">
                  Thinking...
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* INPUT */}
        <div className="border-t border-gray-200 bg-white px-6 py-4">
          <form
            onSubmit={handleSubmit}
            className="max-w-3xl mx-auto flex items-center gap-3"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about sales, customers, or recommendations..."
              className="flex-1 border border-gray-300 rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              disabled={isLoading}
            />

            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-3 rounded-lg flex items-center justify-center disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </div>
    </DashboardLayout>
  );
}