"use client";

import React from "react";

interface AttributionEntry {
  asset: string;
  feature: string;
  importance: number;
}


interface InterpretabilitySectionProps {
  attributions: AttributionEntry[];
  logs?: string[];
}

const InterpretabilitySection: React.FC<InterpretabilitySectionProps> = ({ attributions, logs }) => {
  return (
    <section className="bg-primary text-emerald-50 rounded-card p-md shadow-2xl relative overflow-hidden border border-emerald-900/50">
      <div className="absolute top-0 right-0 p-6 opacity-10 pointer-events-none">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="120"
          height="120"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8z" />
          <path d="M12 6v6l4 2" />
        </svg>
      </div>

      <div className="flex flex-col gap-6 relative z-10">
        <div>
          <h2 className="text-h3 font-h3 mb-1 flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-secondary-container animate-pulse" />
            Granular Model Attribution
          </h2>
          <p className="text-sm text-emerald-300/60 font-body">
            Explainable AI (XAI) diagnostics for current optimization vector.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-lg">
          {/* Feature Importance List */}
          <div className="space-y-md">
            <h3 className="text-xs font-bold uppercase tracking-widest text-emerald-400/80">
              Primary SHAP Drivers
            </h3>
            <div className="space-y-4">
              {attributions && attributions.length > 0 ? (
                attributions.map((item, idx) => (
                  <div key={`${item.asset}-${item.feature}-${idx}`} className="group">
                    <div className="flex justify-between items-center text-[10px] font-bold uppercase mb-1.5 text-emerald-400/70 group-hover:text-emerald-300 transition-colors">
                      <span>{item.asset} - {item.feature}</span>
                      <span className={item.importance > 0 ? "text-emerald-400" : "text-secondary-container"}>
                        {item.importance > 0 ? "+" : ""}{item.importance.toFixed(4)}
                      </span>
                    </div>
                    <div className="h-2 w-full bg-emerald-950/50 rounded-full overflow-hidden border border-emerald-900/30">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ease-out ${
                          item.importance > 0 ? "bg-emerald-400" : "bg-secondary-container"
                        }`}
                        style={{ width: `${Math.min(Math.abs(item.importance) * 20, 1) * 100}%` }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-[10px] text-emerald-500/40 italic">Waiting for model inference...</p>
              )}
            </div>
          </div>

          {/* Granular Insights & System Logs */}
          <div className="flex flex-col gap-lg">
            <div className="space-y-md">
              <h3 className="text-xs font-bold uppercase tracking-widest text-emerald-400/80">
                Decision Attribution
              </h3>
              <div className="bg-black/30 rounded-xl p-4 border border-emerald-900/30 space-y-4">
                <div className="flex items-start gap-3">
                  <div className="p-1.5 bg-emerald-900/50 rounded text-emerald-400">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-xs font-bold text-white mb-0.5">Asset-Level Contribution</p>
                    <p className="text-[11px] text-emerald-300/60 leading-relaxed">
                      Allocation driven by high cross-sectional momentum and low covariance with VIX spikes.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-3">
                  <div className="p-1.5 bg-secondary-container/20 rounded text-secondary-container">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                      <line x1="12" y1="9" x2="12" y2="13" />
                      <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-xs font-bold text-white mb-0.5">Tail Risk Suppression</p>
                    <p className="text-[11px] text-emerald-300/60 leading-relaxed">
                      LSTM hidden state identifies emerging yield curve inversion. Model is actively suppressing risk-heavy exposure.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-black/20 p-sm rounded-lg border border-emerald-900/20">
              {logs && logs.length > 0 ? (
                logs.map((log, i) => (
                  <p key={i} className="font-mono text-[10px] text-emerald-500/80 leading-relaxed">
                    <span className="text-white">SYS_LOG:</span> {log}
                  </p>
                ))
              ) : (
                <p className="font-mono text-[10px] text-emerald-500/80 leading-relaxed italic">
                  Awaiting system report...
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default InterpretabilitySection;

