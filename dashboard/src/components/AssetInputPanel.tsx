"use client";

import React, { useState } from "react";

interface AssetInput {
  ticker: string;
  expectedReturn: number;
}

interface AssetInputPanelProps {
  onOptimize: (assets: Record<string, number>, vix: number) => void;
  isLoading: boolean;
}

const AssetInputPanel: React.FC<AssetInputPanelProps> = ({ onOptimize, isLoading }) => {
  const [assets, setAssets] = useState<AssetInput[]>([
    { ticker: "MSFT", expectedReturn: 0.12 },
    { ticker: "AAPL", expectedReturn: 0.10 },
    { ticker: "GOOGL", expectedReturn: 0.08 },
    { ticker: "AMZN", expectedReturn: 0.15 },
  ]);
  const [vix, setVix] = useState<number>(18.5);

  const addAsset = () => {
    setAssets([...assets, { ticker: "", expectedReturn: 0.05 }]);
  };

  const removeAsset = (index: number) => {
    setAssets(assets.filter((_, i) => i !== index));
  };

  const updateAsset = (index: number, field: keyof AssetInput, value: string | number) => {
    const newAssets = [...assets];
    if (field === "ticker") {
      newAssets[index].ticker = (value as string).toUpperCase();
    } else {
      newAssets[index].expectedReturn = value as number;
    }
    setAssets(newAssets);
  };

  const handleOptimize = () => {
    const assetMap: Record<string, number> = {};
    assets.forEach((a) => {
      if (a.ticker) assetMap[a.ticker] = a.expectedReturn / 100; // Convert % to decimal
    });
    onOptimize(assetMap, vix);
  };

  return (
    <div className="glass-card rounded-card p-6 border border-outline-variant/30 h-full flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-h3 text-primary">Configure Portfolio</h3>
          <p className="text-foreground/50 text-xs font-medium">Add assets and set your market outlook</p>
        </div>
        <button
          onClick={addAsset}
          className="p-2 bg-primary/5 hover:bg-primary/10 text-primary rounded-lg transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
        </button>
      </div>

      {/* Asset List */}
      <div className="flex-1 overflow-y-auto space-y-3 mb-6 pr-2 custom-scrollbar">
        {assets.map((asset, index) => (
          <div key={index} className="flex items-center gap-3 bg-surface-container/30 p-3 rounded-xl border border-outline-variant/10 group">
            <div className="flex-1">
              <label className="text-[10px] font-bold text-foreground/40 uppercase mb-1 block">Ticker</label>
              <input
                type="text"
                value={asset.ticker}
                onChange={(e) => updateAsset(index, "ticker", e.target.value)}
                placeholder="e.g. BTC"
                className="w-full bg-transparent border-none text-sm font-bold text-primary focus:ring-0 p-0"
              />
            </div>
            <div className="w-24">
              <label className="text-[10px] font-bold text-foreground/40 uppercase mb-1 block">Exp. Return %</label>
              <input
                type="number"
                step="0.1"
                value={isNaN(asset.expectedReturn) ? "" : asset.expectedReturn * 100}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  updateAsset(index, "expectedReturn", isNaN(val) ? 0 : val / 100);
                }}
                className="w-full bg-transparent border-none text-sm font-mono font-bold text-secondary-container focus:ring-0 p-0"
              />
            </div>
            <button
              onClick={() => removeAsset(index)}
              className="opacity-0 group-hover:opacity-100 p-2 text-foreground/20 hover:text-red-500 transition-all"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
          </div>
        ))}
      </div>

      {/* Market Sentiment */}
      <div className="pt-4 border-t border-outline-variant/20 space-y-4">
        <div>
          <div className="flex justify-between items-center mb-2">
            <label className="text-xs font-bold text-primary uppercase tracking-wider">Market Fear (VIX)</label>
            <span className={`text-xs font-mono font-bold ${vix > 30 ? "text-red-500" : "text-emerald-500"}`}>
              {vix.toFixed(1)}
            </span>
          </div>
          <input
            type="range"
            min="10"
            max="60"
            step="0.5"
            value={vix}
            onChange={(e) => setVix(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-surface-container rounded-lg appearance-none cursor-pointer accent-primary"
          />
          <div className="flex justify-between mt-1 text-[9px] font-bold text-foreground/30 uppercase">
            <span>Calm</span>
            <span>Normal</span>
            <span>Panic</span>
          </div>
        </div>

        <button
          onClick={handleOptimize}
          disabled={isLoading || assets.length === 0}
          className="w-full py-4 bg-primary text-white rounded-2xl font-bold shadow-lg shadow-primary/20 hover:bg-primary/90 transition-all active:scale-[0.98] disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          ) : (
            <>
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
              Optimize Portfolio
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default AssetInputPanel;
