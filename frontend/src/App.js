import { useState } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Menu, X } from "lucide-react";
import { Toaster } from "@/components/ui/sonner";
import { Sidebar, StatusBar } from "@/components/Nav";
import Overview from "@/pages/Overview";
import Explore from "@/pages/Explore";
import Features from "@/pages/Features";
import Experiments from "@/pages/Experiments";
import Results from "@/pages/Results";
import BackendLog from "@/pages/BackendLog";
import Report from "@/pages/Report";
import "@/App.css";

const Brand = () => (
  <div className="px-3 mb-8">
    <p className="font-serif text-lg font-bold text-slate-900 leading-tight">OULAD<br />At-Risk Lab</p>
    <p className="mt-1 text-[10px] font-mono uppercase tracking-widest text-slate-500">Learning analytics notebook</p>
  </div>
);

export default function App() {
  const [open, setOpen] = useState(false);
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-[#FBF9F5] text-slate-800 flex">
        <aside className="hidden lg:flex w-60 flex-shrink-0 border-r border-[#E2DCD2] bg-[#F4EFE8] h-screen sticky top-0 flex-col p-4">
          <Brand />
          <Sidebar />
          <p className="mt-auto px-3 text-[10px] text-slate-400 leading-relaxed">Data: Kuzilek et al. (2017), OULAD, CC-BY 4.0</p>
        </aside>
        {open && (
          <div className="fixed inset-0 z-50 lg:hidden">
            <div className="absolute inset-0 bg-slate-900/30" onClick={() => setOpen(false)} />
            <aside className="relative w-64 h-full bg-[#F4EFE8] p-4 border-r border-[#E2DCD2]">
              <button data-testid="mobile-nav-close" className="absolute right-3 top-3 p-1" onClick={() => setOpen(false)}><X className="w-5 h-5" /></button>
              <Brand />
              <Sidebar onNavigate={() => setOpen(false)} />
            </aside>
          </div>
        )}
        <div className="flex-1 min-w-0 flex flex-col">
          <div className="bg-white/95 backdrop-blur border-b border-[#E2DCD2] px-4 sm:px-8 py-3 flex items-center justify-between sticky top-0 z-40">
            <button data-testid="mobile-nav-open" className="lg:hidden p-1 -ml-1" onClick={() => setOpen(true)}><Menu className="w-5 h-5" /></button>
            <p className="hidden sm:block text-xs text-slate-500">Predicting students at risk of failing or withdrawing · midterm cut-off</p>
            <StatusBar />
          </div>
          <main className="px-4 sm:px-8 lg:px-12 py-8 lg:py-10 max-w-[1400px] w-full">
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/explore" element={<Explore />} />
              <Route path="/features" element={<Features />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/results" element={<Results />} />
              <Route path="/logs" element={<BackendLog />} />
              <Route path="/report" element={<Report />} />
            </Routes>
          </main>
        </div>
      </div>
      <Toaster position="bottom-right" />
    </BrowserRouter>
  );
}
