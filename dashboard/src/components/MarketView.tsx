"use client";

import React from "react";

interface RegimeData {
  dominant: string;
  probabilities: Record<string, number>;
}

interface RiskFlagsData {
  vix_current: number;
  vix_override_active: boolean;
  vix_threshold: number;
}

interface MarketViewProps {
  regime: RegimeData | null;
  riskFlags: RiskFlagsData | null;
  allocations: { ticker: string; weight: number }[];
}

const SECTORS = [
  { name: "Technology", tickers: ["MSFT", "GOOGL", "META", "AMZN"], change: 1.24, sentiment: "bullish" },
  { name: "Energy", tickers: ["XOM"], change: -0.38, sentiment: "neutral" },
  { name: "Healthcare", tickers: ["JNJ", "UNH"], change: 0.67, sentiment: "bullish" },
  { name: "Financials", tickers: ["JPM", "BAC"], change: -0.12, sentiment: "bearish" },
  { name: "Consumer", tickers: ["PG", "KO"], change: 0.45, sentiment: "neutral" },
  { name: "Industrials", tickers: ["CAT", "BA"], change: 0.89, sentiment: "bullish" },
];

const MACRO_INDICATORS = [
  { name: "10Y Treasury", value: "4.32%", delta: "+3bp", direction: "up" as const },
  { name: "DXY Index", value: "104.8", delta: "-0.2%", direction: "down" as const },
  { name: "Gold Spot", value: "$2,341", delta: "+0.8%", direction: "up" as const },
  { name: "Crude WTI", value: "$78.4", delta: "-1.2%", direction: "down" as const },
  { name: "Fed Funds Rate", value: "5.25%", delta: "0bp", direction: "flat" as const },
  { name: "CPI YoY", value: "3.1%", delta: "-0.2%", direction: "down" as const },
];

const MarketView: React.FC<MarketViewProps> = ({ regime, riskFlags, allocations }) => {
  const regimeColor = (key: string) => {
    if (regime?.dominant === key) return "ring-2 ring-secondary-container";
    return "";
  };

  return (
    <div className="space-y-6">
      {/* Regime Probability Matrix */}
      <div className="glass-card rounded-card p-md border border-outline-variant/30">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h3 className="text-h3 text-primary">Market Regime Detection</h3>
            <p className="text-foreground/50 text-sm font-medium">Hidden Markov Model state probabilities</p>
          </div>
          <span className={`px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest ${
            regime?.dominant === "bull" ? "bg-emerald-100 text-emerald-700" :
            regime?.dominant === "bear" ? "bg-red-100 text-red-700" :
            "bg-amber-100 text-amber-700"
          }`}>
            {regime?.dominant || "UNKNOWN"} REGIME
          </span>
        </div>

        <div className="grid grid-cols-3 gap-4 mb-6">
          {[
            { key: "bull", label: "Bullish Expansion", icon: "↗", color: "emerald" },
            { key: "neutral", label: "Sideways Consolidation", icon: "→", color: "amber" },
            { key: "bear", label: "Bearish Contraction", icon: "↘", color: "red" },
          ].map((r) => {
            const prob = regime?.probabilities?.[r.key] || 0;
            const isActive = regime?.dominant === r.key;
            return (
              <div
                key={r.key}
                className={`p-5 rounded-2xl border transition-all ${
                  isActive
                    ? "bg-primary text-white border-primary shadow-lg shadow-primary/20"
                    : "bg-surface-container border-outline-variant/20 hover:border-outline-variant/40"
                } ${regimeColor(r.key)}`}
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-2xl">{r.icon}</span>
                  {isActive && (
                    <span className="w-2 h-2 rounded-full bg-secondary-container animate-pulse" />
                  )}
                </div>
                <p className={`text-xs font-bold uppercase tracking-wider mb-1 ${isActive ? "text-white/70" : "text-foreground/40"}`}>
                  {r.label}
                </p>
                <p className={`text-2xl font-h2 ${isActive ? "text-white" : "text-primary"}`}>
                  {(prob * 100).toFixed(1)}%
                </p>
                {/* Mini probability bar */}
                <div className={`h-1 w-full rounded-full mt-3 ${isActive ? "bg-white/20" : "bg-outline-variant/30"}`}>
                  <div
                    className={`h-full rounded-full transition-all duration-1000 ${
                      isActive ? "bg-secondary-container" : r.color === "emerald" ? "bg-emerald-500" : r.color === "red" ? "bg-red-400" : "bg-amber-400"
                    }`}
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>

        {/* VIX Monitor */}
        <div className="bg-surface-container rounded-2xl p-5 border border-outline-variant/20">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${
                riskFlags?.vix_override_active ? "bg-secondary-container animate-pulse" : "bg-emerald-500"
              }`} />
              <span className="text-sm font-bold text-primary">VIX Volatility Index</span>
            </div>
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
              riskFlags?.vix_override_active
                ? "bg-secondary-container/10 text-secondary-container"
                : "bg-emerald-100 text-emerald-700"
            }`}>
              {riskFlags?.vix_override_active ? "CIRCUIT BREAKER ACTIVE" : "NORMAL RANGE"}
            </span>
          </div>
          <div className="flex items-end gap-6">
            <div>
              <p className="text-4xl font-h1 text-primary">{riskFlags?.vix_current?.toFixed(1) || "—"}</p>
              <p className="text-[10px] text-foreground/40 uppercase tracking-widest font-bold mt-1">CURRENT LEVEL</p>
            </div>
            <div className="flex-1">
              <div className="h-3 bg-outline-variant/20 rounded-full relative overflow-hidden">
                <div
                  className="absolute left-0 top-0 h-full rounded-full transition-all duration-700 bg-gradient-to-r from-emerald-500 via-amber-400 to-red-500"
                  style={{ width: `${Math.min((riskFlags?.vix_current || 0) / 60 * 100, 100)}%` }}
                />
                {/* Threshold marker */}
                <div
                  className="absolute top-0 h-full w-0.5 bg-secondary-container"
                  style={{ left: `${(riskFlags?.vix_threshold || 30) / 60 * 100}%` }}
                />
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[9px] text-foreground/30 font-bold">0</span>
                <span className="text-[9px] text-secondary-container font-bold">THRESHOLD: {riskFlags?.vix_threshold || 30}</span>
                <span className="text-[9px] text-foreground/30 font-bold">60</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Sector Heatmap + Macro Indicators */}
      <div className="grid grid-cols-12 gap-6">
        {/* Sector Performance */}
        <div className="col-span-7 glass-card rounded-card p-md border border-outline-variant/30">
          <h3 className="text-h3 text-primary mb-1">Sector Heatmap</h3>
          <p className="text-foreground/50 text-sm font-medium mb-6">Cross-sector momentum signals</p>
          <div className="grid grid-cols-3 gap-3">
            {SECTORS.map((sector) => {
              const isPositive = sector.change >= 0;
              return (
                <div
                  key={sector.name}
                  className={`p-4 rounded-xl border transition-all hover:scale-[1.02] cursor-pointer ${
                    isPositive
                      ? "bg-emerald-50/80 border-emerald-200/50 hover:border-emerald-300"
                      : "bg-red-50/80 border-red-200/50 hover:border-red-300"
                  }`}
                >
                  <p className="text-xs font-bold text-primary mb-1">{sector.name}</p>
                  <p className={`text-lg font-h2 ${isPositive ? "text-emerald-700" : "text-red-600"}`}>
                    {isPositive ? "+" : ""}{sector.change}%
                  </p>
                  <p className="text-[9px] font-bold uppercase tracking-widest mt-1 text-foreground/30">
                    {sector.tickers.join(" · ")}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Macro Indicators */}
        <div className="col-span-5 glass-card rounded-card p-md border border-outline-variant/30">
          <h3 className="text-h3 text-primary mb-1">Macro Indicators</h3>
          <p className="text-foreground/50 text-sm font-medium mb-6">Global economic pulse</p>
          <div className="space-y-0">
            {MACRO_INDICATORS.map((ind) => (
              <div
                key={ind.name}
                className="flex items-center justify-between py-3.5 border-b border-outline-variant/15 last:border-0 group hover:bg-surface-container/50 transition-colors px-2 -mx-2 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className={`w-1.5 h-1.5 rounded-full ${
                    ind.direction === "up" ? "bg-emerald-500" :
                    ind.direction === "down" ? "bg-red-400" : "bg-foreground/20"
                  }`} />
                  <span className="text-sm font-medium text-foreground/70">{ind.name}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm font-bold text-primary">{ind.value}</span>
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                    ind.direction === "up" ? "text-emerald-600 bg-emerald-50" :
                    ind.direction === "down" ? "text-red-500 bg-red-50" :
                    "text-foreground/40 bg-surface-container"
                  }`}>
                    {ind.delta}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketView;
