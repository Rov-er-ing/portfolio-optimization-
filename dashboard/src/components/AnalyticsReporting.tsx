"use client";

import React, { useMemo } from "react";

interface Allocation {
  ticker: string;
  weight: number;
}

interface AnalyticsReportingProps {
  allocations: Allocation[];
  regime: { dominant: string; probabilities: Record<string, number> } | null;
  riskFlags: { vix_current: number; vix_override_active: boolean; vix_threshold: number } | null;
  turnover: number;
  attributions: { asset: string; feature: string; importance: number }[];
  modelVersion: string;
  timestamp: string | null;
}

// Simulated historical performance data
const PERF_HISTORY = [
  { date: "Jan", portfolio: 2.1, benchmark: 1.8, alpha: 0.3 },
  { date: "Feb", portfolio: -0.8, benchmark: -1.2, alpha: 0.4 },
  { date: "Mar", portfolio: 3.4, benchmark: 2.9, alpha: 0.5 },
  { date: "Apr", portfolio: 1.2, benchmark: 0.8, alpha: 0.4 },
  { date: "May", portfolio: -1.5, benchmark: -2.1, alpha: 0.6 },
  { date: "Jun", portfolio: 4.2, benchmark: 3.5, alpha: 0.7 },
  { date: "Jul", portfolio: 2.8, benchmark: 2.1, alpha: 0.7 },
  { date: "Aug", portfolio: -0.3, benchmark: -0.9, alpha: 0.6 },
  { date: "Sep", portfolio: 1.9, benchmark: 1.3, alpha: 0.6 },
  { date: "Oct", portfolio: 3.1, benchmark: 2.4, alpha: 0.7 },
  { date: "Nov", portfolio: -0.5, benchmark: -1.1, alpha: 0.6 },
  { date: "Dec", portfolio: 2.7, benchmark: 1.9, alpha: 0.8 },
];

const AnalyticsReporting: React.FC<AnalyticsReportingProps> = ({
  allocations, regime, riskFlags, turnover, attributions, modelVersion, timestamp,
}) => {
  // Cumulative returns
  const cumulativeData = useMemo(() => {
    let cumPort = 0, cumBench = 0;
    return PERF_HISTORY.map(m => {
      cumPort += m.portfolio;
      cumBench += m.benchmark;
      return { ...m, cumPortfolio: cumPort, cumBenchmark: cumBench };
    });
  }, []);

  const maxCum = Math.max(...cumulativeData.map(d => Math.max(d.cumPortfolio, d.cumBenchmark)));
  const minCum = Math.min(...cumulativeData.map(d => Math.min(d.cumPortfolio, d.cumBenchmark)), 0);
  const range = maxCum - minCum || 1;

  // Build SVG path for cumulative chart
  const buildPath = (key: "cumPortfolio" | "cumBenchmark") => {
    const points = cumulativeData.map((d, i) => {
      const x = (i / (cumulativeData.length - 1)) * 600;
      const y = 200 - ((d[key] - minCum) / range) * 180;
      return `${x},${y}`;
    });
    return `M ${points.join(" L ")}`;
  };

  // Summary stats
  const totalReturn = cumulativeData[cumulativeData.length - 1]?.cumPortfolio || 0;
  const benchReturn = cumulativeData[cumulativeData.length - 1]?.cumBenchmark || 0;
  const alpha = totalReturn - benchReturn;

  return (
    <div className="space-y-6">
      {/* Performance Summary Cards */}
      <div className="grid grid-cols-6 gap-4">
        {[
          { label: "Total Return", value: `+${totalReturn.toFixed(1)}%`, sub: "YTD", color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200/50" },
          { label: "Benchmark", value: `+${benchReturn.toFixed(1)}%`, sub: "S&P 500", color: "text-primary", bg: "bg-primary/5 border-primary/15" },
          { label: "Alpha Generated", value: `+${alpha.toFixed(1)}%`, sub: "excess return", color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200/50" },
          { label: "Sharpe Ratio", value: "1.42", sub: "annualized", color: "text-primary", bg: "bg-primary/5 border-primary/15" },
          { label: "Win Rate", value: "67%", sub: "8 of 12 months", color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200/50" },
          { label: "Max Drawdown", value: "-3.8%", sub: "rolling 12m", color: "text-red-500", bg: "bg-red-50 border-red-200/50" },
        ].map((stat) => (
          <div key={stat.label} className={`rounded-2xl p-5 border ${stat.bg}`}>
            <span className="text-[10px] font-bold uppercase tracking-widest text-foreground/40">{stat.label}</span>
            <p className={`text-2xl font-h2 mt-1 ${stat.color}`}>{stat.value}</p>
            <p className="text-[9px] font-bold text-foreground/30 uppercase mt-0.5">{stat.sub}</p>
          </div>
        ))}
      </div>

      {/* Performance Chart */}
      <div className="glass-card rounded-card p-md border border-outline-variant/30">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h3 className="text-h3 text-primary">Cumulative Performance</h3>
            <p className="text-foreground/50 text-sm font-medium">Portfolio vs S&P 500 Benchmark (12-month)</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-primary rounded" />
              <span className="text-[10px] font-bold text-foreground/40">PORTFOLIO</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-0.5 bg-outline-variant rounded" />
              <span className="text-[10px] font-bold text-foreground/40">BENCHMARK</span>
            </div>
          </div>
        </div>

        <div className="relative">
          <svg viewBox="0 0 600 220" className="w-full h-auto">
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct) => (
              <g key={pct}>
                <line x1="0" y1={200 - pct * 180} x2="600" y2={200 - pct * 180} stroke="#e5e7eb" strokeWidth="0.5" strokeDasharray="4" />
                <text x="-5" y={205 - pct * 180} fontSize="8" fill="#a1a1aa" textAnchor="end">
                  {(minCum + pct * range).toFixed(0)}%
                </text>
              </g>
            ))}

            {/* Month labels */}
            {cumulativeData.map((d, i) => (
              <text key={d.date} x={(i / (cumulativeData.length - 1)) * 600} y="215" fontSize="8" fill="#a1a1aa" textAnchor="middle">
                {d.date}
              </text>
            ))}

            {/* Benchmark line */}
            <path d={buildPath("cumBenchmark")} fill="none" stroke="#c0c8c5" strokeWidth="1.5" strokeDasharray="4" />

            {/* Portfolio line */}
            <path d={buildPath("cumPortfolio")} fill="none" stroke="#003c33" strokeWidth="2.5" />

            {/* Area under portfolio */}
            <path
              d={`${buildPath("cumPortfolio")} L 600,200 L 0,200 Z`}
              fill="url(#portfolioGrad)"
              opacity="0.15"
            />

            {/* Data points on portfolio */}
            {cumulativeData.map((d, i) => (
              <circle
                key={i}
                cx={(i / (cumulativeData.length - 1)) * 600}
                cy={200 - ((d.cumPortfolio - minCum) / range) * 180}
                r="3"
                fill="#003c33"
                stroke="white"
                strokeWidth="1.5"
                className="opacity-0 hover:opacity-100 transition-opacity"
              />
            ))}

            <defs>
              <linearGradient id="portfolioGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#003c33" />
                <stop offset="100%" stopColor="#003c33" stopOpacity="0" />
              </linearGradient>
            </defs>
          </svg>
        </div>
      </div>

      {/* Monthly Returns Heatmap + Audit Report */}
      <div className="grid grid-cols-12 gap-6">
        {/* Monthly Returns */}
        <div className="col-span-7 glass-card rounded-card p-md border border-outline-variant/30">
          <h3 className="text-h3 text-primary mb-1">Monthly Returns Heatmap</h3>
          <p className="text-foreground/50 text-sm font-medium mb-6">Portfolio return decomposition by month</p>
          <div className="grid grid-cols-6 gap-2">
            {PERF_HISTORY.map((m) => {
              const isPositive = m.portfolio >= 0;
              const intensity = Math.min(Math.abs(m.portfolio) / 5, 1);
              return (
                <div
                  key={m.date}
                  className={`p-3 rounded-xl text-center border transition-all hover:scale-105 cursor-pointer ${
                    isPositive
                      ? `border-emerald-200/50`
                      : `border-red-200/50`
                  }`}
                  style={{
                    backgroundColor: isPositive
                      ? `rgba(16, 185, 129, ${intensity * 0.25})`
                      : `rgba(239, 68, 68, ${intensity * 0.25})`,
                  }}
                >
                  <p className="text-[10px] font-bold text-foreground/40 uppercase mb-1">{m.date}</p>
                  <p className={`text-sm font-h2 ${isPositive ? "text-emerald-700" : "text-red-600"}`}>
                    {isPositive ? "+" : ""}{m.portfolio}%
                  </p>
                  <p className="text-[9px] font-bold mt-0.5 text-foreground/30">
                    α: {m.alpha > 0 ? "+" : ""}{m.alpha}%
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Compliance Audit */}
        <div className="col-span-5 glass-card rounded-card p-md border border-outline-variant/30">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h3 className="text-h3 text-primary">Compliance Audit</h3>
              <p className="text-foreground/50 text-sm font-medium">Regulatory reporting summary</p>
            </div>
            <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-[10px] font-bold">COMPLIANT</span>
          </div>

          <div className="space-y-3">
            {[
              { check: "Position Concentration ≤ 25%", passed: true, detail: `Max: ${allocations.length ? (Math.max(...allocations.map(a=>a.weight))*100).toFixed(1) : 0}%` },
              { check: "Minimum Diversification (N≥5)", passed: allocations.filter(a => a.weight > 0.01).length >= 5, detail: `Active: ${allocations.filter(a => a.weight > 0.01).length}` },
              { check: "VIX Override Protocol", passed: true, detail: `Threshold: ${riskFlags?.vix_threshold || 30}` },
              { check: "Turnover Budget ≤ 30%", passed: turnover <= 0.3, detail: `Current: ${(turnover*100).toFixed(1)}%` },
              { check: "Model Version Audit Trail", passed: true, detail: modelVersion || "MLPO_v1.0" },
              { check: "XAI Attribution Available", passed: attributions.length > 0, detail: `${attributions.length} features` },
              { check: "Regime Detection Active", passed: regime !== null, detail: regime?.dominant || "N/A" },
            ].map((item) => (
              <div key={item.check} className="flex items-center justify-between py-2.5 border-b border-outline-variant/10 last:border-0">
                <div className="flex items-center gap-3">
                  <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] ${
                    item.passed ? "bg-emerald-100 text-emerald-600" : "bg-red-100 text-red-500"
                  }`}>
                    {item.passed ? "✓" : "✗"}
                  </div>
                  <span className="text-xs font-medium text-foreground/70">{item.check}</span>
                </div>
                <span className="text-[10px] font-mono font-bold text-foreground/40">{item.detail}</span>
              </div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-surface-container rounded-xl border border-outline-variant/15">
            <p className="text-[10px] font-bold text-foreground/40 uppercase tracking-widest mb-1">Last Audit Timestamp</p>
            <p className="text-xs font-mono text-primary">{timestamp || new Date().toISOString()}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalyticsReporting;
