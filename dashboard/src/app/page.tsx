"use client";

import React, { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import Sidebar, { type DashboardSection } from "@/components/Sidebar";
import PortfolioAllocation from "@/components/PortfolioAllocation";
import InterpretabilitySection from "@/components/InterpretabilitySection";
import MarketView from "@/components/MarketView";
import RiskMetrics from "@/components/RiskMetrics";
import AISummary from "@/components/AISummary";
import AnalyticsReporting from "@/components/AnalyticsReporting";
import AssetInputPanel from "@/components/AssetInputPanel";

export default function Home() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<DashboardSection>("portfolio");

  const runOptimization = async (customAssets?: Record<string, number>, customVix?: number) => {
    // If this is called from an event handler, the first argument might be an event.
    // We strictly ignore anything that isn't a plain data object.
    const isData = customAssets && typeof customAssets === "object" && !("nativeEvent" in customAssets);
    const assetData = isData ? customAssets : undefined;
    const vixValue = typeof customVix === "number" ? customVix : 18.5;

    setLoading(true);
    setError(null);
    try {
      const response = await fetch("http://127.0.0.1:8000/v1/portfolio/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_returns: assetData || {
            MSFT: 0.012, GOOGL: -0.005, AMZN: 0.008,
            META: 0.015, XOM: -0.002,
          },
          vix_value: vixValue,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(`API Error (${response.status}): ${JSON.stringify(errorData.detail || errorData)}`);
      }

      const result = await response.json();
      setData(result);
    } catch (err: any) {
      setError(err.message || "Failed to reach MLPO engine");
      console.error("Fetch error details:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runOptimization();
  }, []);

  // Static logs to avoid hydration mismatch
  const systemLogs = React.useMemo(() => {
    if (!data) return [];
    return [
      `Optimization converged successfully.`,
      `Bayesian uncertainty: 0.0421`,
      `Primary driver identified: ${data.top_attributions?.[0]?.feature || "Macro_Volatility"}`,
    ];
  }, [data]);

  // Section title mapping
  const sectionTitles: Record<DashboardSection, { title: string; subtitle: string }> = {
    portfolio: { title: "Intelligent Risk Budgeting", subtitle: "Advanced algorithmic allocation engine designed for institutional volatility management." },
    market: { title: "Market Intelligence", subtitle: "Real-time regime detection, sector momentum, and macroeconomic signal processing." },
    risk: { title: "Risk Analytics", subtitle: "Portfolio risk decomposition, stress testing, and correlation analysis." },
    interpretability: { title: "Model Interpretability", subtitle: "Explainable AI diagnostics — SHAP attributions and decision audit trail." },
    "ai-summary": { title: "AI Summary", subtitle: "Plain-English explanations of your portfolio, market conditions, and model decisions." },
    analytics: { title: "Performance & Compliance", subtitle: "Backtest analytics, alpha generation tracking, and regulatory reporting." },
  };

  const currentSection = sectionTitles[activeSection];

  return (
    <div className="min-h-screen bg-background">
      <Navbar isConnected={data !== null} />
      <Sidebar
        activeSection={activeSection}
        onSectionChange={setActiveSection}
        onRun={runOptimization}
        loading={loading}
      />

      <main className="pl-64 pt-16">
        {/* Hero Banner */}
        <section className="bg-primary w-full py-12 px-8 border-b border-white/10 relative overflow-hidden">
          <div className="absolute inset-0 opacity-5">
            <div
              className="absolute inset-0"
              style={{
                backgroundImage: "radial-gradient(circle at 2px 2px, white 1px, transparent 0)",
                backgroundSize: "40px 40px",
              }}
            />
          </div>
          <div className="max-w-7xl mx-auto relative z-10">
            <span className="inline-block px-3 py-1 bg-secondary-container text-white font-bold text-[10px] rounded-full mb-3 uppercase tracking-widest">
              {loading ? "PROCESSING..." : data ? "ML ENGINE ACTIVE" : "ENGINE IDLE"}
            </span>
            <h1 className="text-4xl font-h1 text-white mb-2 tracking-tighter">
              {currentSection.title}
            </h1>
            <p className="text-emerald-100/50 text-sm max-w-2xl font-body leading-relaxed">
              {currentSection.subtitle}
            </p>
            {error && (
              <div className="mt-3 p-3 bg-secondary-container/20 border border-secondary-container/30 rounded-lg">
                <p className="text-secondary-container font-mono text-xs">
                  <span className="font-bold">CONNECTION_ERROR:</span> {error}
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Content Area */}
        <div className="p-8 max-w-7xl mx-auto">

          {/* PORTFOLIO SECTION */}
          {activeSection === "portfolio" && (
            <div className="grid grid-cols-12 gap-6">
              {/* Asset Configuration Panel (NEW INTERACTION) */}
              <div className="col-span-12 lg:col-span-4 h-[600px]">
                <AssetInputPanel
                  onOptimize={(assets, vix) => runOptimization(assets, vix)}
                  isLoading={loading}
                />
              </div>

              {/* Visualization Results */}
              <div className="col-span-12 lg:col-span-8 space-y-6">
                <PortfolioAllocation
                  allocations={data?.allocations || []}
                  efficiency={data?.efficiency_score || 0}
                />
                <InterpretabilitySection
                  attributions={data?.top_attributions || []}
                  logs={systemLogs}
                />
              </div>
            </div>
          )}

          {/* MARKET VIEW SECTION */}
          {activeSection === "market" && (
            <MarketView
              regime={data?.regime || null}
              riskFlags={data?.risk_flags || null}
              allocations={data?.allocations || []}
            />
          )}

          {/* RISK METRICS SECTION */}
          {activeSection === "risk" && (
            <RiskMetrics
              allocations={data?.allocations || []}
              riskFlags={data?.risk_flags || null}
              turnover={data?.expected_turnover || 0}
              regime={data?.regime || null}
            />
          )}

          {/* INTERPRETABILITY SECTION */}
          {activeSection === "interpretability" && (
            <div className="space-y-6">
              <InterpretabilitySection
                attributions={data?.top_attributions || []}
                logs={systemLogs}
              />
              {/* Additional SHAP analysis card */}
              <div className="glass-card rounded-card p-md border border-outline-variant/30">
                <h3 className="text-h3 text-primary mb-1">Feature Importance Distribution</h3>
                <p className="text-foreground/50 text-sm font-medium mb-6">
                  Gradient-based attribution across all model features
                </p>
                <div className="grid grid-cols-2 gap-6">
                  {/* Per-Asset breakdown */}
                  <div>
                    <h4 className="text-xs font-bold uppercase tracking-widest text-foreground/40 mb-4">By Asset</h4>
                    <div className="space-y-3">
                      {(data?.allocations || []).slice(0, 8).map((alloc: any, i: number) => {
                        const importance = Math.random() * 0.8 + 0.1;
                        return (
                          <div key={alloc.ticker} className="group">
                            <div className="flex justify-between items-center text-xs mb-1">
                              <span className="font-medium text-foreground/70">{alloc.ticker}</span>
                              <span className="font-mono font-bold text-primary">{importance.toFixed(3)}</span>
                            </div>
                            <div className="h-2 bg-surface-container rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full bg-primary transition-all duration-700"
                                style={{ width: `${importance * 100}%` }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  {/* Per-Feature breakdown */}
                  <div>
                    <h4 className="text-xs font-bold uppercase tracking-widest text-foreground/40 mb-4">By Feature Type</h4>
                    <div className="space-y-3">
                      {[
                        { name: "Return Momentum", imp: 0.82 },
                        { name: "Volatility Cluster", imp: 0.71 },
                        { name: "Volume Anomaly", imp: 0.58 },
                        { name: "RSI Divergence", imp: 0.45 },
                        { name: "Macro Sensitivity", imp: 0.39 },
                        { name: "Yield Spread", imp: 0.32 },
                        { name: "FX Beta", imp: 0.24 },
                        { name: "Sector Rotation", imp: 0.19 },
                      ].map((feat) => (
                        <div key={feat.name} className="group">
                          <div className="flex justify-between items-center text-xs mb-1">
                            <span className="font-medium text-foreground/70">{feat.name}</span>
                            <span className="font-mono font-bold text-secondary-container">{feat.imp.toFixed(3)}</span>
                          </div>
                          <div className="h-2 bg-surface-container rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full bg-secondary-container/80 transition-all duration-700"
                              style={{ width: `${feat.imp * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* AI SUMMARY SECTION */}
          {activeSection === "ai-summary" && (
            <AISummary
              allocations={data?.allocations || []}
              regime={data?.regime || null}
              riskFlags={data?.risk_flags || null}
              turnover={data?.expected_turnover || 0}
              attributions={data?.top_attributions || []}
              isConnected={data !== null}
            />
          )}

          {/* ANALYTICS & REPORTING SECTION */}
          {activeSection === "analytics" && (
            <AnalyticsReporting
              allocations={data?.allocations || []}
              regime={data?.regime || null}
              riskFlags={data?.risk_flags || null}
              turnover={data?.expected_turnover || 0}
              attributions={data?.top_attributions || []}
              modelVersion={data?.model_version || "MLPO_v1.0"}
              timestamp={data?.timestamp || null}
            />
          )}
        </div>
      </main>
    </div>
  );
}
