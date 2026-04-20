import { useState, useRef, useEffect } from "react";
import { API } from "../services/api";
import DashboardLayout from "../layouts/DashboardLayout.jsx";
import { Send, Bot, User, Clock, AlertTriangle, Sparkles, Zap, ShieldCheck } from "lucide-react";

export default function AIAgent() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Welcome to your Retail Intelligence Hub. I am synchronized with your market data and ready to extract insights. How can I assist your business strategy today?",
    }
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

    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const res = await API.aiAgent(userMessage);

      if (res.status === "error") {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `System Error: ${res.message}`, isError: true },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { 
            role: "assistant", 
            content: res.response,
            time_taken: res.time_taken 
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Connection interrupted. Please verify your network and ensure the intelligence server is operational.", isError: true },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <DashboardLayout>
      <div className="flex flex-col h-[calc(100vh-4rem)] bg-[#050510] relative overflow-hidden">
        
        {/* Background Decorative Elements */}
        <div className="absolute top-[-10%] right-[-10%] w-[500px] h-[500px] bg-indigo-600/10 rounded-full blur-[120px] pointer-events-none"></div>
        <div className="absolute bottom-[-10%] left-[-10%] w-[400px] h-[400px] bg-purple-600/10 rounded-full blur-[100px] pointer-events-none"></div>

        {/* AI Header */}
        <header className="flex items-center justify-between px-8 py-6 border-b border-white/5 bg-white/[0.02] backdrop-blur-xl z-20">
          <div className="flex items-center gap-5">
            <div className="relative">
              <div className="w-14 h-14 bg-gradient-to-tr from-indigo-500 via-purple-500 to-pink-500 rounded-2xl flex items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.4)]">
                <Bot className="w-8 h-8 text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-emerald-500 border-2 border-[#050510] rounded-full"></div>
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
                MinersAI <span className="text-indigo-400 font-normal text-sm px-2 py-0.5 bg-indigo-500/10 rounded-md border border-indigo-500/20">v2.0</span>
              </h1>
              <div className="flex items-center gap-3 mt-1">
                <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-emerald-400 font-bold bg-emerald-400/5 px-2 py-0.5 rounded">
                  <Zap className="w-3 h-3" /> Live Analysis
                </span>
                <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-indigo-400 font-bold bg-indigo-400/5 px-2 py-0.5 rounded">
                    <ShieldCheck className="w-3 h-3" /> Secure Data
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Chat Window */}
        <div className="flex-1 overflow-y-auto px-6 py-10 space-y-10 scrollbar-thin scrollbar-thumb-white/10">
          <div className="max-w-4xl mx-auto space-y-10">
            {messages.map((msg, idx) => {
              const isUser = msg.role === "user";
              return (
                <div key={idx} className={`flex gap-6 ${isUser ? "flex-row-reverse" : "flex-row"} animate-in fade-in slide-in-from-bottom-4 duration-500`}>
                  
                  {/* Avatar Icons */}
                  <div className={`shrink-0 w-12 h-12 rounded-2xl flex items-center justify-center shadow-lg transition-transform hover:scale-110 ${
                    isUser 
                      ? "bg-white/5 border border-white/10 text-white" 
                      : msg.isError 
                        ? "bg-red-500/20 border border-red-500/30 text-red-400" 
                        : "bg-indigo-600 border border-indigo-500 text-white"
                  }`}>
                    {isUser ? <User className="w-6 h-6" /> : msg.isError ? <AlertTriangle className="w-6 h-6" /> : <Sparkles className="w-6 h-6" />}
                  </div>

                  {/* Message Bubble */}
                  <div className={`relative max-w-[85%] md:max-w-[75%] px-6 py-5 rounded-3xl group transition-all ${
                    isUser 
                      ? "bg-white/[0.03] border border-white/10 text-indigo-100 rounded-tr-none" 
                      : msg.isError
                        ? "bg-red-500/5 text-red-200 border border-red-500/20 rounded-tl-none"
                        : "bg-gradient-to-br from-indigo-600/10 to-purple-600/5 text-gray-100 border border-white/5 rounded-tl-none backdrop-blur-md shadow-2xl"
                  }`}>
                    <div className="text-[17px] leading-relaxed font-medium">
                      {msg.content}
                    </div>
                    
                    {/* Time taken indicator for AI responses */}
                    {!isUser && msg.time_taken && (
                      <div className="mt-4 flex items-center gap-2 text-[11px] font-bold text-white/30 uppercase tracking-widest border-t border-white/5 pt-3">
                        <Clock className="w-3.5 h-3.5" />
                        Analysis complete in {msg.time_taken}s
                      </div>
                    )}
                  </div>
                </div>
              );
            })}

            {/* AI Thinking Animation */}
            {isLoading && (
              <div className="flex gap-6">
                <div className="shrink-0 w-12 h-12 rounded-2xl bg-indigo-600 border border-indigo-500 flex items-center justify-center shadow-lg animate-pulse">
                  <Sparkles className="w-6 h-6 text-white" />
                </div>
                <div className="bg-white/5 border border-white/10 rounded-3xl rounded-tl-none px-6 py-5 flex items-center gap-3">
                  <div className="flex space-x-1.5 h-6 items-center">
                    <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1.5 h-1.5 bg-pink-400 rounded-full animate-bounce"></div>
                  </div>
                  <span className="text-xs font-bold text-white/40 uppercase tracking-widest">Processing Data...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Terminal */}
        <div className="p-8 z-30">
          <div className="max-w-4xl mx-auto">
            <div className="relative group bg-white/[0.03] backdrop-blur-2xl border border-white/10 rounded-3xl p-2.5 shadow-2xl transition-all focus-within:border-indigo-500/50 focus-within:ring-4 focus-within:ring-indigo-500/10">
              <form onSubmit={handleSubmit} className="flex items-center gap-3">
                <div className="flex-1 px-5">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Describe the retail insight you're looking for..."
                    className="w-full bg-transparent border-none text-white placeholder-white/30 focus:ring-0 py-4 text-lg font-medium"
                    disabled={isLoading}
                  />
                </div>
                <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="bg-gradient-to-tr from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white h-14 w-14 rounded-2xl flex items-center justify-center transition-all shadow-[0_10px_20px_rgba(79,70,229,0.3)] hover:shadow-[0_15px_30px_rgba(79,70,229,0.5)] active:scale-95 disabled:opacity-30 disabled:grayscale disabled:cursor-not-allowed group"
                >
                  <Send className="w-6 h-6 transition-transform group-hover:translate-x-1 group-hover:-translate-y-1" />
                </button>
              </form>
            </div>
            <p className="text-center mt-5 text-[11px] text-white/20 font-bold uppercase tracking-[0.2em]">
              AI System powered by Advanced Retrieval Reasoning
            </p>
          </div>
        </div>

      </div>
    </DashboardLayout>
  );
}
