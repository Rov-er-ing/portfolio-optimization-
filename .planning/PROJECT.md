# PROJECT: ML-Powered Portfolio Optimization System (MLPO v1.0)

## What This Is
MLPO v1.0 is an end-to-end differentiable portfolio optimization pipeline designed for local consumer GPU environments. It integrates deep learning volatility forecasting with convex optimization layers (`cvxpylayers`) to allow gradients to flow from portfolio loss metrics back to feature extraction layers.

## Core Value
The system provides a mathematically rigorous bridge between predictive modeling (LSTM/GMM) and modern portfolio theory (Convex Optimization), enabling the direct training of models to optimize for final financial outcomes (Sharpe Ratio, Drawdown) rather than just proxy errors (MSE).

## Context
- **Target Hardware:** 16GB RAM, 4-6GB VRAM (NVIDIA RTX 3050 class).
- **Asset Universe:** 45 instruments (30 Equities, 10 Fixed Income, 5 Commodities) + 6 Macro State Variables.
- **Constraint:** PSD covariance matrix enforcement via Cholesky factor prediction.

## Requirements

### Validated
(None yet — ship to validate)

### Active
- [ ] **Module A: LSTM Volatility Forecaster** — 3-layer LSTM with Cholesky factor output.
- [ ] **Module B: Regime Detection** — GMM + LSTM encoder with VIX-based tail-risk override.
- [ ] **Module C: Risk Budgeting Layer** — `cvxpylayers` integration for differentiable optimization.
- [ ] **Module D: Sparse Attention** — Linear complexity $\mathcal{O}(pk)$ attention mechanism.
- [ ] **Data Pipeline** — Lazy loading, memory-mapped tensors, and AMP (FP16) support.
- [ ] **Backtest Engine** — Walk-forward simulation with transaction costs and financial metrics (Calmar, VaR).
- [ ] **Interpretability** — DeepSHAP and KernelSHAP integration for regulatory-grade audits.
- [ ] **API Layer** — FastAPI with Pydantic v2 validation.

### Out of Scope
- [ ] Full Kubernetes Deployment — Reserved for future versions.
- [ ] Real-time Execution/Trading — Initial version focus is on backtesting and research.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| `cvxpylayers` | Enables gradient flow through the optimizer | — Pending |
| Cholesky Factorization | Guarantees PSD covariance matrices | — Pending |
| AMP (FP16) | Necessary for training on 4GB-6GB VRAM | — Pending |
| Sparse Attention | Maintains scalability with asset universe size | — Pending |

## Evolution
This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-25 after initialization*
