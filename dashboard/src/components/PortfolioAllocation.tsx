"use client";

import React from "react";

interface Allocation {
  ticker: string;
  weight: number;
}

interface PortfolioAllocationProps {
  allocations: Allocation[];
  efficiency?: number;
}

const colorMap: Record<string, string> = {
  MSFT: "#003c33",
  GOOGL: "#fc7557",
  AMZN: "#9dd1c4",
  META: "#dadade",
  XOM: "#74a79b",
};

const PortfolioAllocation: React.FC<PortfolioAllocationProps> = ({ allocations, efficiency = 78 }) => {
  const displayAllocations = allocations.slice(0, 5);

  return (
    <div className="glass-card rounded-card p-md border border-outline-variant/30">
      <div className="flex justify-between items-center mb-lg">
        <div>
          <h3 className="text-h3 text-primary">Portfolio Allocation</h3>
          <p className="text-foreground/50 text-sm font-medium">Real-time Asset Distribution</p>
        </div>
        <div className="flex gap-2">
          <span className="px-3 py-1 bg-surface-container-high rounded-full text-[10px] font-bold">AUTO-REBALANCE: ON</span>
          <span className={`px-3 py-1 rounded-full text-[10px] font-bold ${allocations.length > 0 ? "bg-primary-container text-white" : "bg-surface-container text-foreground/40"}`}>
            {allocations.length > 0 ? "STABLE" : "IDLE"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-lg items-center">
        <div className="relative aspect-square max-w-[240px] mx-auto flex items-center justify-center">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" fill="none" stroke="#f3f3f7" strokeWidth="10" />
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="#003c33"
              strokeWidth="10"
              strokeDasharray="251.2"
              strokeDashoffset={251.2 * (1 - efficiency / 100)}
              className="transition-all duration-1000 ease-in-out"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <span className="text-3xl font-h2 text-primary">{efficiency.toFixed(2)}%</span>
            <span className="text-[10px] uppercase tracking-widest text-foreground/40 font-bold">Efficiency</span>
          </div>
        </div>

        <div className="space-y-2">
          {displayAllocations.length > 0 ? (
            displayAllocations.map((asset) => (
              <div
                key={asset.ticker}
                className="flex items-center justify-between p-3 rounded-xl hover:bg-white/40 transition-all border border-transparent hover:border-outline-variant/20"
              >
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: colorMap[asset.ticker] || "#707976" }} />
                  <span className="text-sm font-medium text-foreground">{asset.ticker}</span>
                </div>
                <span className="font-mono text-sm font-bold text-primary">{(asset.weight * 100).toFixed(2)}%</span>
              </div>
            ))
          ) : (
            <p className="text-xs text-foreground/40 italic text-center py-8">Waiting for optimization signal...</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioAllocation;

