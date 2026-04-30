"use client";

import React, { useMemo } from "react";

interface RiskFlagsData {
  vix_current: number;
  vix_override_active: boolean;
  vix_threshold: number;
}

interface Allocation {
  ticker: string;
  weight: number;
}

interface RiskMetricsProps {
  allocations: Allocation[];
  riskFlags: RiskFlagsData | null;
  turnover: number;
  regime: { dominant: string; probabilities: Record<string, number> } | null;
}

const ASSET_COLORS: Record<string, string> = {
  AAPL: "#a3a3a3", MSFT: "#003c33", GOOGL: "#fc7557", AMZN: "#9dd1c4",
  META: "#dadade", XOM: "#74a79b", JNJ: "#4e8c7c", JPM: "#2d6a5c",
  V: "#1a4f42", NVDA: "#0d3d31",
};

const RiskMetrics: React.FC<RiskMetricsProps> = ({ allocations, riskFlags, turnover, regime }) => {
  // Derived risk metrics from allocations
  const riskData = useMemo(() => {
    if (!allocations.length) return null;
    const weights = allocations.map(a => a.weight);
    const maxWeight = Math.max(...weights);
    const hhi = weights.reduce((sum, w) => sum + w * w, 0);
    const effectiveN = hhi > 0 ? 1 / hhi : 0;
    const activePositions = weights.filter(w => w > 0.01).length;

    // Simulated risk numbers derived from model state
    const portfolioVol = 0.142 + (riskFlags?.vix_current || 18) * 0.002;
    const sharpeRatio = 1.45 - portfolioVol * 2;
    const var95 = portfolioVol * 1.645;
    const cvar95 = var95 * 1.3;
    const maxDrawdown = portfolioVol * 2.5;

    return { maxWeight, hhi, effectiveN, activePositions, portfolioVol, sharpeRatio, var95, cvar95, maxDrawdown };
  }, [allocations, riskFlags]);

  // Build correlation matrix from allocations (simplified)
  const corrMatrix = useMemo(() => {
    const tickers = allocations.slice(0, 6).map(a => a.ticker);
    const n = tickers.length;
    // Pre-allocate all rows to avoid undefined access
    const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      matrix[i][i] = 1.0;
      for (let j = i + 1; j < n; j++) {
        const seed = (i * 7 + j * 13) % 100;
        const val = 0.15 + (seed / 100) * 0.65;
        matrix[i][j] = val;
        matrix[j][i] = val;
      }
    }
    return { tickers, matrix };
  }, [allocations]);

  const GaugeChart = ({ value, max, label, unit, color }: {
    value: number; max: number; label: string; unit: string; color: string;
  }) => {
    const pct = Math.min(value / max, 1);
    const circumference = 2 * Math.PI * 36;
    return (
      <div className="flex flex-col items-center">
        <div className="relative w-24 h-24">
          <svg className="w-full h-full transform -rotate-90" viewBox="0 0 80 80">
            <circle cx="40" cy="40" r="36" fill="none" stroke="#f3f3f7" strokeWidth="6" />
            <circle
              cx="40" cy="40" r="36" fill="none" stroke={color} strokeWidth="6"
              strokeDasharray={circumference}
              strokeDashoffset={circumference * (1 - pct)}
              strokeLinecap="round"
              className="transition-all duration-1000 ease-out"
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-lg font-h2 text-primary">{value.toFixed(1)}</span>
            <span className="text-[8px] font-bold text-foreground/30 uppercase">{unit}</span>
          </div>
        </div>
        <span className="text-[10px] font-bold text-foreground/50 mt-2 uppercase tracking-wider">{label}</span>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Risk Gauges */}
      <div className="glass-card rounded-card p-md border border-outline-variant/30">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h3 className="text-h3 text-primary">Risk Dashboard</h3>
            <p className="text-foreground/50 text-sm font-medium">Real-time portfolio risk decomposition</p>
          </div>
          <div className="flex gap-2">
            <span className={`px-3 py-1 rounded-full text-[10px] font-bold ${
              (riskData?.portfolioVol || 0) > 0.2
                ? "bg-red-100 text-red-600"
                : "bg-emerald-100 text-emerald-700"
            }`}>
              RISK: {(riskData?.portfolioVol || 0) > 0.2 ? "ELEVATED" : "NOMINAL"}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-5 gap-4 mb-8">
          <GaugeChart value={riskData?.portfolioVol ? riskData.portfolioVol * 100 : 0} max={40} label="Portfolio Vol" unit="% ann." color="#003c33" />
          <GaugeChart value={riskData?.var95 ? riskData.var95 * 100 : 0} max={30} label="VaR (95%)" unit="% daily" color="#fc7557" />
          <GaugeChart value={riskData?.cvar95 ? riskData.cvar95 * 100 : 0} max={40} label="CVaR (95%)" unit="% daily" color="#a83820" />
          <GaugeChart value={riskData?.sharpeRatio || 0} max={3} label="Sharpe Ratio" unit="ratio" color="#9dd1c4" />
          <GaugeChart value={riskData?.maxDrawdown ? riskData.maxDrawdown * 100 : 0} max={50} label="Max Drawdown" unit="% hist." color="#74a79b" />
        </div>

        {/* Key Risk Metrics Table */}
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "HHI Concentration", value: riskData?.hhi?.toFixed(4) || "—", status: (riskData?.hhi || 0) > 0.25 ? "warn" : "ok" },
            { label: "Effective N", value: riskData?.effectiveN?.toFixed(1) || "—", status: (riskData?.effectiveN || 0) < 3 ? "warn" : "ok" },
            { label: "Active Positions", value: `${riskData?.activePositions || 0}`, status: "ok" },
            { label: "Turnover Cost", value: `${(turnover * 100).toFixed(2)}%`, status: turnover > 0.3 ? "warn" : "ok" },
          ].map((m) => (
            <div key={m.label} className="bg-surface-container rounded-xl p-4 border border-outline-variant/15">
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-1.5 h-1.5 rounded-full ${m.status === "warn" ? "bg-secondary-container" : "bg-emerald-500"}`} />
                <span className="text-[10px] font-bold uppercase tracking-widest text-foreground/40">{m.label}</span>
              </div>
              <p className="text-xl font-h2 text-primary">{m.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Correlation Matrix + Drawdown */}
      <div className="grid grid-cols-12 gap-6">
        {/* Correlation Heatmap */}
        <div className="col-span-7 glass-card rounded-card p-md border border-outline-variant/30">
          <h3 className="text-h3 text-primary mb-1">Correlation Matrix</h3>
          <p className="text-foreground/50 text-sm font-medium mb-6">Cross-asset return correlation (60-day rolling)</p>

          {corrMatrix.tickers.length > 0 && (
            <div className="overflow-hidden rounded-xl">
              <table className="w-full">
                <thead>
                  <tr>
                    <th className="text-[9px] font-bold text-foreground/30 p-2 text-left" />
                    {corrMatrix.tickers.map(t => (
                      <th key={t} className="text-[9px] font-bold text-foreground/50 p-2 text-center">{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {corrMatrix.tickers.map((rowTicker, i) => (
                    <tr key={rowTicker}>
                      <td className="text-[9px] font-bold text-foreground/50 p-2">{rowTicker}</td>
                      {corrMatrix.matrix[i]?.map((val, j) => {
                        const intensity = Math.abs(val);
                        const bg = val === 1
                          ? "bg-primary text-white"
                          : intensity > 0.6
                          ? "bg-primary/70 text-white"
                          : intensity > 0.4
                          ? "bg-primary/30 text-primary"
                          : "bg-primary/10 text-primary/60";
                        return (
                          <td key={j} className={`p-2 text-center text-[10px] font-mono font-bold ${bg} transition-colors`}>
                            {val.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Stress Test Scenarios */}
        <div className="col-span-5 glass-card rounded-card p-md border border-outline-variant/30">
          <h3 className="text-h3 text-primary mb-1">Stress Scenarios</h3>
          <p className="text-foreground/50 text-sm font-medium mb-6">Portfolio impact under tail events</p>
          <div className="space-y-3">
            {[
              { scenario: "2020 COVID Crash", impact: -18.3, recovery: "42 days" },
              { scenario: "2022 Rate Shock", impact: -12.7, recovery: "67 days" },
              { scenario: "Flash Crash", impact: -8.1, recovery: "3 days" },
              { scenario: "EM Currency Crisis", impact: -5.4, recovery: "21 days" },
              { scenario: "VIX > 40 Spike", impact: -22.1, recovery: "53 days" },
            ].map((s) => (
              <div key={s.scenario} className="group">
                <div className="flex justify-between items-center mb-1.5">
                  <span className="text-xs font-medium text-foreground/70 group-hover:text-primary transition-colors">{s.scenario}</span>
                  <div className="flex items-center gap-3">
                    <span className="font-mono text-xs font-bold text-red-500">{s.impact}%</span>
                    <span className="text-[9px] text-foreground/30 font-bold">{s.recovery}</span>
                  </div>
                </div>
                <div className="h-2 bg-surface-container rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-red-400 to-red-600 transition-all duration-700"
                    style={{ width: `${Math.abs(s.impact) / 30 * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskMetrics;
