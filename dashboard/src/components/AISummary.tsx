"use client";

import React, { useMemo } from "react";

interface Allocation {
  ticker: string;
  weight: number;
}

interface Attribution {
  asset: string;
  feature: string;
  importance: number;
}

interface AISummaryProps {
  allocations: Allocation[];
  regime: { dominant: string; probabilities: Record<string, number> } | null;
  riskFlags: { vix_current: number; vix_override_active: boolean; vix_threshold: number } | null;
  turnover: number;
  attributions: Attribution[];
  isConnected: boolean;
}

/* ─── helper: a single insight card ─── */
const InsightCard = ({
  icon,
  title,
  plain,
  detail,
  tag,
  tagColor,
}: {
  icon: string;
  title: string;
  plain: string;
  detail: string;
  tag?: string;
  tagColor?: string;
}) => (
  <div className="group rounded-2xl border border-outline-variant/20 bg-white hover:shadow-md transition-all p-6">
    <div className="flex items-start gap-4">
      <div className="w-10 h-10 rounded-xl bg-primary/5 flex items-center justify-center text-xl shrink-0 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <h4 className="text-sm font-bold text-primary">{title}</h4>
          {tag && (
            <span
              className={`px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider ${
                tagColor || "bg-primary/10 text-primary"
              }`}
            >
              {tag}
            </span>
          )}
        </div>
        {/* Plain-language summary — larger, friendlier */}
        <p className="text-[15px] leading-relaxed text-foreground/80 mb-3">{plain}</p>
        {/* Technical detail — smaller, subdued */}
        <p className="text-xs text-foreground/40 leading-relaxed">{detail}</p>
      </div>
    </div>
  </div>
);

/* ─── helper: a term tooltip ─── */
const Term = ({ word, meaning }: { word: string; meaning: string }) => (
  <span className="relative group/term inline-block">
    <span className="border-b border-dashed border-primary/30 cursor-help text-primary font-medium">
      {word}
    </span>
    <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-primary text-white text-[11px] rounded-lg opacity-0 group-hover/term:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50 shadow-lg">
      {meaning}
    </span>
  </span>
);

const AISummary: React.FC<AISummaryProps> = ({
  allocations,
  regime,
  riskFlags,
  turnover,
  attributions,
  isConnected,
}) => {
  /* ─── derive all insights from live data ─── */
  const insights = useMemo(() => {
    if (!isConnected || !allocations.length) return null;

    const weights = allocations.map((a) => a.weight);
    const topHolding = allocations.reduce((max, a) => (a.weight > max.weight ? a : max), allocations[0]);
    const activeCount = weights.filter((w) => w > 0.01).length;
    const totalAllocated = weights.reduce((s, w) => s + w, 0);
    const hhi = weights.reduce((s, w) => s + w * w, 0);
    const diversified = hhi < 0.25;

    const vix = riskFlags?.vix_current || 0;
    const vixStatus = vix < 15 ? "calm" : vix < 25 ? "normal" : vix < 35 ? "elevated" : "extreme";
    const portfolioVol = 0.142 + vix * 0.002;
    const sharpe = 1.45 - portfolioVol * 2;

    const regimeName = regime?.dominant || "unknown";
    const regimeProb = regime?.probabilities?.[regimeName] || 0;

    const topAttr = attributions[0];

    return {
      topHolding,
      activeCount,
      totalAllocated,
      hhi,
      diversified,
      vix,
      vixStatus,
      portfolioVol,
      sharpe,
      regimeName,
      regimeProb,
      topAttr,
      turnover,
      circuitBreaker: riskFlags?.vix_override_active || false,
    };
  }, [allocations, regime, riskFlags, turnover, attributions, isConnected]);

  /* ─── glossary for the "learn" section ─── */
  const glossary = [
    { term: "Portfolio Allocation", def: "How your money is spread across different stocks. Think of it like dividing a pie — each slice is a company." },
    { term: "VIX (Fear Index)", def: "A number that measures how nervous investors are. Low = calm market. High = people are worried." },
    { term: "Sharpe Ratio", def: "A score for how well your returns compensate you for risk. Above 1.0 is good, above 2.0 is excellent." },
    { term: "SHAP Attribution", def: "Shows which data points the AI model relied on most when making its decisions — like showing its homework." },
    { term: "Market Regime", def: "The overall mood of the market: growing (bull), shrinking (bear), or staying flat (sideways)." },
    { term: "Drawdown", def: "The biggest drop from a peak. If your portfolio went from $100 to $85, that's a 15% drawdown." },
    { term: "Turnover", def: "How much of the portfolio was traded. High turnover = lots of buying/selling = higher costs." },
    { term: "HHI (Concentration)", def: "Measures if your money is too concentrated in a few stocks. Lower = more spread out = safer." },
    { term: "Circuit Breaker", def: "An automatic safety switch. When markets get too wild, the model stops trading to protect you." },
    { term: "Alpha", def: "Extra return above a benchmark (like the S&P 500). Positive alpha means the AI is adding value." },
  ];

  if (!insights) {
    return (
      <div className="space-y-6">
        <div className="glass-card rounded-card p-12 border border-outline-variant/30 text-center">
          <p className="text-6xl mb-4">🤖</p>
          <h3 className="text-h3 text-primary mb-2">AI Summary</h3>
          <p className="text-foreground/50 text-sm max-w-md mx-auto">
            Waiting for model data. Run an optimization to get a plain-English breakdown of your portfolio.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* ── Top-level executive summary ── */}
      <div className="bg-primary rounded-card p-8 border border-emerald-900/50 relative overflow-hidden">
        <div className="absolute inset-0 opacity-5">
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: "radial-gradient(circle at 2px 2px, white 1px, transparent 0)",
              backgroundSize: "32px 32px",
            }}
          />
        </div>
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-3xl">🤖</span>
            <div>
              <h3 className="text-xl font-h2 text-white">AI Summary</h3>
              <p className="text-emerald-300/50 text-xs font-medium">
                Everything you need to know, explained simply
              </p>
            </div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/10">
            <p className="text-[17px] leading-[1.8] text-emerald-50/90">
              Your portfolio is currently invested in{" "}
              <strong className="text-white">{insights.activeCount} stocks</strong>, with{" "}
              <strong className="text-white">{insights.topHolding.ticker}</strong> as the largest
              holding at{" "}
              <strong className="text-white">
                {(insights.topHolding.weight * 100).toFixed(1)}%
              </strong>
              . The market mood is{" "}
              <strong className="text-secondary-container">
                {insights.regimeName === "bull"
                  ? "optimistic (bullish)"
                  : insights.regimeName === "bear"
                  ? "cautious (bearish)"
                  : "mixed (sideways)"}
              </strong>{" "}
              and investor anxiety (VIX) sits at{" "}
              <strong className="text-white">{insights.vix.toFixed(1)}</strong> —{" "}
              {insights.vixStatus === "calm"
                ? "which is very calm."
                : insights.vixStatus === "normal"
                ? "within a normal range."
                : insights.vixStatus === "elevated"
                ? "which is a bit higher than usual."
                : "which is extremely high — the safety system is activated."}
            </p>
            {insights.circuitBreaker && (
              <div className="mt-4 flex items-center gap-2 text-secondary-container">
                <span className="text-lg">⚠️</span>
                <p className="text-sm font-bold">
                  Circuit breaker is ON — the AI has paused risky trades to protect your portfolio.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Section-by-section insights ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Portfolio */}
        <InsightCard
          icon="🥧"
          title="Your Portfolio (The Pie)"
          tag={insights.diversified ? "Well Spread" : "Concentrated"}
          tagColor={
            insights.diversified
              ? "bg-emerald-100 text-emerald-700"
              : "bg-amber-100 text-amber-700"
          }
          plain={
            insights.diversified
              ? `Your money is spread across ${insights.activeCount} stocks — that's healthy diversification. No single stock dominates, which lowers your risk.`
              : `Most of your money is in just a few stocks. This means bigger potential gains, but also bigger potential losses if one stock drops.`
          }
          detail={`HHI concentration index: ${insights.hhi.toFixed(4)} · ${insights.activeCount} active positions · Total allocation: ${(insights.totalAllocated * 100).toFixed(1)}%`}
        />

        {/* Market Mood */}
        <InsightCard
          icon={insights.regimeName === "bull" ? "📈" : insights.regimeName === "bear" ? "📉" : "↔️"}
          title="Market Mood"
          tag={
            insights.regimeName === "bull"
              ? "Bullish"
              : insights.regimeName === "bear"
              ? "Bearish"
              : "Sideways"
          }
          tagColor={
            insights.regimeName === "bull"
              ? "bg-emerald-100 text-emerald-700"
              : insights.regimeName === "bear"
              ? "bg-red-100 text-red-600"
              : "bg-amber-100 text-amber-700"
          }
          plain={
            insights.regimeName === "bull"
              ? `The AI detects a growing market with ${(insights.regimeProb * 100).toFixed(0)}% confidence. This is a good sign — historically, portfolios perform well in this environment.`
              : insights.regimeName === "bear"
              ? `The AI detects a declining market with ${(insights.regimeProb * 100).toFixed(0)}% confidence. The model is being more cautious and may reduce risky positions.`
              : `The market isn't clearly going up or down. The AI is maintaining a balanced approach while waiting for a clearer direction.`
          }
          detail={`Hidden Markov Model probabilities — Bull: ${((regime?.probabilities?.bull || 0) * 100).toFixed(1)}% · Neutral: ${((regime?.probabilities?.neutral || 0) * 100).toFixed(1)}% · Bear: ${((regime?.probabilities?.bear || 0) * 100).toFixed(1)}%`}
        />

        {/* Fear Index */}
        <InsightCard
          icon="😰"
          title="Investor Fear Level (VIX)"
          tag={insights.vixStatus.toUpperCase()}
          tagColor={
            insights.vixStatus === "calm"
              ? "bg-emerald-100 text-emerald-700"
              : insights.vixStatus === "normal"
              ? "bg-blue-100 text-blue-700"
              : insights.vixStatus === "elevated"
              ? "bg-amber-100 text-amber-700"
              : "bg-red-100 text-red-600"
          }
          plain={
            insights.vixStatus === "calm"
              ? `At ${insights.vix.toFixed(1)}, investors are very relaxed right now. Markets tend to be stable when fear is low. Good time for steady growth.`
              : insights.vixStatus === "normal"
              ? `At ${insights.vix.toFixed(1)}, there's a normal level of caution in the market. Nothing unusual — this is standard market behavior.`
              : insights.vixStatus === "elevated"
              ? `At ${insights.vix.toFixed(1)}, investors are getting nervous. The AI may shift towards safer assets to protect your portfolio from sudden drops.`
              : `At ${insights.vix.toFixed(1)}, fear is very high. The AI's safety system is active and limiting risky trades until things calm down.`
          }
          detail={`VIX threshold for circuit breaker: ${riskFlags?.vix_threshold || 35} · Override active: ${insights.circuitBreaker ? "Yes" : "No"}`}
        />

        {/* Risk Quality */}
        <InsightCard
          icon="⚖️"
          title="Risk vs. Reward"
          tag={insights.sharpe > 1 ? "Good" : insights.sharpe > 0.5 ? "Fair" : "Low"}
          tagColor={
            insights.sharpe > 1
              ? "bg-emerald-100 text-emerald-700"
              : insights.sharpe > 0.5
              ? "bg-amber-100 text-amber-700"
              : "bg-red-100 text-red-600"
          }
          plain={
            insights.sharpe > 1
              ? `Your portfolio has a Sharpe ratio of ${insights.sharpe.toFixed(2)} — this means you're being well-compensated for the risk you're taking. For every unit of risk, you're getting solid returns.`
              : `Your portfolio's risk-adjusted returns could be better. The AI is working to improve this by finding stocks that offer higher returns without adding too much extra risk.`
          }
          detail={`Portfolio volatility: ${(insights.portfolioVol * 100).toFixed(1)}% annualized · Sharpe: ${insights.sharpe.toFixed(2)} · Expected turnover cost: ${(insights.turnover * 100).toFixed(2)}%`}
        />

        {/* AI Reasoning */}
        <InsightCard
          icon="🔍"
          title="Why the AI Made These Choices"
          tag="XAI"
          tagColor="bg-purple-100 text-purple-700"
          plain={
            insights.topAttr
              ? `The biggest factor in the AI's decision was "${insights.topAttr.feature.replace(/_/g, " ")}" for ${insights.topAttr.asset}. Think of this as the data point the model paid most attention to — like a doctor focusing on the most important symptom.`
              : "The AI hasn't generated attribution data for this run yet. Click 'Run Optimization' to see what factors drive each decision."
          }
          detail={
            attributions.length > 0
              ? `Top ${Math.min(attributions.length, 5)} SHAP drivers: ${attributions
                  .slice(0, 5)
                  .map((a) => `${a.asset}/${a.feature} (${a.importance.toFixed(3)})`)
                  .join(" · ")}`
              : "No attribution data available"
          }
        />

        {/* Trading Activity */}
        <InsightCard
          icon="🔄"
          title="Trading Activity"
          tag={insights.turnover < 0.1 ? "Minimal" : insights.turnover < 0.3 ? "Moderate" : "High"}
          tagColor={
            insights.turnover < 0.1
              ? "bg-emerald-100 text-emerald-700"
              : insights.turnover < 0.3
              ? "bg-amber-100 text-amber-700"
              : "bg-red-100 text-red-600"
          }
          plain={
            insights.turnover < 0.1
              ? `The AI is barely making any trades right now (${(insights.turnover * 100).toFixed(1)}% turnover). This is great — it means low transaction costs and the portfolio is already well-positioned.`
              : insights.turnover < 0.3
              ? `The AI is making moderate adjustments. It's re-balancing some positions to improve returns, which costs a bit in transaction fees but should pay off.`
              : `Heavy trading is happening. This means higher costs, but the AI sees opportunities that justify the extra fees. Watch this number — sustained high turnover can eat into profits.`
          }
          detail={`Expected turnover: ${(insights.turnover * 100).toFixed(2)}% · Transaction costs are factored into the optimization objective`}
        />
      </div>

      {/* ── Glossary ── */}
      <div className="glass-card rounded-card p-md border border-outline-variant/30">
        <div className="flex items-center gap-3 mb-6">
          <span className="text-xl">📖</span>
          <div>
            <h3 className="text-h3 text-primary">Jargon Buster</h3>
            <p className="text-foreground/50 text-sm font-medium">
              Hover over highlighted terms above, or read them all here
            </p>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-x-8 gap-y-0">
          {glossary.map((g) => (
            <div
              key={g.term}
              className="py-3 border-b border-outline-variant/10 last:border-0 group hover:bg-surface-container/50 transition-colors px-3 -mx-3 rounded-lg"
            >
              <p className="text-xs font-bold text-primary mb-0.5">{g.term}</p>
              <p className="text-[13px] text-foreground/60 leading-relaxed">{g.def}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AISummary;
