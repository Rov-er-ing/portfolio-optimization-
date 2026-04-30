"use client";

import React from "react";

export type DashboardSection = "portfolio" | "market" | "risk" | "interpretability" | "ai-summary" | "analytics";

interface SidebarProps {
  activeSection: DashboardSection;
  onSectionChange: (section: DashboardSection) => void;
  onRun?: () => void;
  loading?: boolean;
}

const MENU_ITEMS: { key: DashboardSection; name: string; icon: string }[] = [
  { key: "portfolio", name: "Portfolio", icon: "📊" },
  { key: "market", name: "Market View", icon: "🌐" },
  { key: "risk", name: "Risk Metrics", icon: "🛡️" },
  { key: "interpretability", name: "Interpretability", icon: "🧠" },
  { key: "ai-summary", name: "AI Summary", icon: "🤖" },
  { key: "analytics", name: "Analytics", icon: "📈" },
];

const Sidebar: React.FC<SidebarProps> = ({ activeSection, onSectionChange, onRun, loading }) => {
  return (
    <aside className="fixed left-0 top-16 h-[calc(100vh-64px)] w-64 bg-white/80 backdrop-blur-[24px] border-r border-outline-variant/30 z-40 flex flex-col">
      <div className="p-8 pb-4">
        <span className="text-[10px] font-bold uppercase tracking-widest text-foreground/40">Main Dashboard</span>
      </div>
      
      <nav className="flex-1 px-4">
        <ul className="space-y-1">
          {MENU_ITEMS.map((item) => (
            <li key={item.key}>
              <button
                onClick={() => onSectionChange(item.key)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all group text-left ${
                  activeSection === item.key
                  ? "bg-primary text-white shadow-lg shadow-primary/20" 
                  : "text-foreground/60 hover:bg-surface-container"
                }`}
              >
                <span className="text-lg">{item.icon}</span>
                {item.name}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="p-4 mt-auto">
        <div className="bg-surface-container rounded-2xl p-6 border border-outline-variant/20">
          <p className="text-xs font-bold text-primary mb-2">Engine Controls</p>
          <p className="text-[10px] text-foreground/50 mb-4 font-medium leading-relaxed">
            Execute manual rebalance signal via Bayesian LSTM.
          </p>
          <button 
            onClick={onRun}
            disabled={loading}
            className="w-full bg-primary text-white text-xs font-bold py-3 rounded-xl hover:bg-primary/90 transition-all active:scale-95 shadow-md shadow-primary/10 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Processing..." : "Run Optimization"}
          </button>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
