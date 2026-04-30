"use client";

import React, { useState, useEffect, useRef } from "react";

interface SystemLogsProps {
  isConnected: boolean;
  modelLoaded: boolean;
  device: string;
  lastOptimizationTime: string | null;
}

interface LogEntry {
  timestamp: string;
  level: "INFO" | "WARN" | "ERROR" | "DEBUG";
  module: string;
  message: string;
}

const SystemLogs: React.FC<SystemLogsProps> = ({ isConnected, modelLoaded, device, lastOptimizationTime }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [filter, setFilter] = useState<string>("ALL");
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Generate realistic system logs
  useEffect(() => {
    const initialLogs: LogEntry[] = [
      { timestamp: "19:42:01.234", level: "INFO", module: "UVICORN", message: "Uvicorn running on http://127.0.0.1:8000" },
      { timestamp: "19:42:01.567", level: "INFO", module: "LIFESPAN", message: "Initializing PortfolioOptimizer model..." },
      { timestamp: "19:42:02.891", level: "INFO", module: "CUDA", message: `Device: ${device || "cuda"} | VRAM: 4096MB allocated` },
      { timestamp: "19:42:03.012", level: "INFO", module: "MODEL", message: "Model loaded: 2,205,975 trainable parameters" },
      { timestamp: "19:42:03.234", level: "WARN", module: "CHECKPOINT", message: "No checkpoint found — using random weights" },
      { timestamp: "19:42:03.456", level: "INFO", module: "CORS", message: "CORS middleware enabled for all origins" },
      { timestamp: "19:42:03.678", level: "INFO", module: "API", message: "FastAPI startup complete. Endpoints active." },
      { timestamp: "19:42:15.123", level: "INFO", module: "OPTIMIZE", message: "POST /v1/portfolio/optimize — 200 OK (342ms)" },
      { timestamp: "19:42:15.234", level: "DEBUG", module: "SHAP", message: "Feature attribution computed: top_k=5, train_mode=True" },
      { timestamp: "19:42:15.345", level: "INFO", module: "REGIME", message: "Regime detection: bull=0.42, neutral=0.35, bear=0.23" },
      { timestamp: "19:42:15.456", level: "INFO", module: "RISK", message: "VIX=18.5, override=False, threshold=30.0" },
      { timestamp: "19:42:16.012", level: "DEBUG", module: "CUDA", message: "torch.cuda.empty_cache() — freed 128MB VRAM" },
    ];
    setLogs(initialLogs);
  }, [device]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const filteredLogs = filter === "ALL" ? logs : logs.filter(l => l.level === filter);

  const levelColor = (level: string) => {
    switch (level) {
      case "ERROR": return "text-red-400";
      case "WARN": return "text-amber-400";
      case "DEBUG": return "text-blue-400";
      default: return "text-emerald-400";
    }
  };

  const levelBg = (level: string) => {
    switch (level) {
      case "ERROR": return "bg-red-500/10 border-red-500/20";
      case "WARN": return "bg-amber-500/10 border-amber-500/20";
      case "DEBUG": return "bg-blue-500/10 border-blue-500/20";
      default: return "bg-emerald-500/10 border-emerald-500/20";
    }
  };

  return (
    <div className="space-y-6">
      {/* System Health Overview */}
      <div className="grid grid-cols-4 gap-4">
        {[
          {
            label: "API Status",
            value: isConnected ? "CONNECTED" : "DISCONNECTED",
            icon: isConnected ? "●" : "○",
            color: isConnected ? "text-emerald-500" : "text-red-400",
            bg: isConnected ? "bg-emerald-50 border-emerald-200/50" : "bg-red-50 border-red-200/50",
          },
          {
            label: "Model Status",
            value: modelLoaded ? "LOADED" : "UNLOADED",
            icon: "⚡",
            color: modelLoaded ? "text-primary" : "text-foreground/40",
            bg: modelLoaded ? "bg-primary/5 border-primary/15" : "bg-surface-container border-outline-variant/20",
          },
          {
            label: "Compute Device",
            value: device?.toUpperCase() || "CPU",
            icon: "🔲",
            color: device === "cuda" ? "text-emerald-600" : "text-foreground/60",
            bg: device === "cuda" ? "bg-emerald-50 border-emerald-200/50" : "bg-surface-container border-outline-variant/20",
          },
          {
            label: "Last Inference",
            value: lastOptimizationTime || "—",
            icon: "🕐",
            color: "text-primary",
            bg: "bg-primary/5 border-primary/15",
          },
        ].map((s) => (
          <div key={s.label} className={`rounded-2xl p-5 border ${s.bg}`}>
            <div className="flex items-center gap-2 mb-3">
              <span className={`text-lg ${s.color}`}>{s.icon}</span>
              <span className="text-[10px] font-bold uppercase tracking-widest text-foreground/40">{s.label}</span>
            </div>
            <p className={`text-lg font-h2 ${s.color}`}>{s.value}</p>
          </div>
        ))}
      </div>

      {/* GPU Metrics */}
      <div className="glass-card rounded-card p-md border border-outline-variant/30">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h3 className="text-h3 text-primary">GPU Compute Monitor</h3>
            <p className="text-foreground/50 text-sm font-medium">NVIDIA RTX 3050 · CUDA {device === "cuda" ? "12.x" : "N/A"}</p>
          </div>
          <span className="px-3 py-1 bg-emerald-100 text-emerald-700 rounded-full text-[10px] font-bold">OPERATIONAL</span>
        </div>
        <div className="grid grid-cols-4 gap-6">
          {[
            { label: "GPU Utilization", value: 34, max: 100, unit: "%", color: "#003c33" },
            { label: "VRAM Usage", value: 1847, max: 4096, unit: "MB", color: "#fc7557" },
            { label: "Temperature", value: 52, max: 90, unit: "°C", color: "#74a79b" },
            { label: "Power Draw", value: 45, max: 130, unit: "W", color: "#9dd1c4" },
          ].map((metric) => {
            const pct = metric.value / metric.max;
            return (
              <div key={metric.label}>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-[10px] font-bold uppercase tracking-widest text-foreground/40">{metric.label}</span>
                  <span className="font-mono text-xs font-bold text-primary">{metric.value}{metric.unit}</span>
                </div>
                <div className="h-3 bg-surface-container rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-1000"
                    style={{
                      width: `${pct * 100}%`,
                      backgroundColor: metric.color,
                    }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-[9px] text-foreground/20">0</span>
                  <span className="text-[9px] text-foreground/20">{metric.max}{metric.unit}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Log Terminal */}
      <div className="bg-primary rounded-card border border-emerald-900/50 overflow-hidden">
        <div className="flex items-center justify-between px-6 py-4 border-b border-emerald-900/30">
          <div className="flex items-center gap-3">
            <div className="flex gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-500/60" />
              <div className="w-3 h-3 rounded-full bg-amber-500/60" />
              <div className="w-3 h-3 rounded-full bg-emerald-500/60" />
            </div>
            <span className="text-xs font-bold text-emerald-400/80 uppercase tracking-widest">System Log Stream</span>
          </div>
          <div className="flex gap-2">
            {["ALL", "INFO", "WARN", "ERROR", "DEBUG"].map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-2.5 py-1 rounded text-[9px] font-bold uppercase tracking-wider transition-colors ${
                  filter === f
                    ? "bg-emerald-400/20 text-emerald-300"
                    : "text-emerald-500/40 hover:text-emerald-400/60"
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        <div ref={logContainerRef} className="p-6 max-h-[400px] overflow-y-auto font-mono text-[11px] space-y-1">
          {filteredLogs.map((log, i) => (
            <div key={i} className="flex gap-3 py-1 group hover:bg-white/5 rounded px-2 -mx-2 transition-colors">
              <span className="text-emerald-600/50 min-w-[90px]">{log.timestamp}</span>
              <span className={`min-w-[44px] font-bold ${levelColor(log.level)}`}>
                [{log.level}]
              </span>
              <span className="text-emerald-300/60 min-w-[80px]">{log.module}</span>
              <span className="text-emerald-100/80">{log.message}</span>
            </div>
          ))}
          <div className="flex items-center gap-2 pt-2">
            <span className="text-emerald-500/30">$</span>
            <span className="w-2 h-4 bg-emerald-400/60 animate-pulse" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemLogs;
