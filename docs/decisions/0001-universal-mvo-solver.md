# ADR-0001: Universal MVO Solver Pivot

## Status
Accepted

## Context
The initial MLPO prototype utilized a "Black Box" LSTM model that mapped directly to a fixed universe of 45 assets. This approach had several institutional-grade flaws:
1.  **Brittleness**: The model failed when users provided fewer than 45 tickers or tickers outside the pre-trained set.
2.  **Constraint Fragility**: Hard-coding allocation logic into model weights made it impossible to enforce legal or institutional constraints (e.g., "minimum 2% allocation per asset").
3.  **Lack of Transparency**: Portfolio Managers could not explain *why* the model allocated 100% to a single asset during high volatility.

We needed a system that is **Universal** (works on any asset list) and **Explainable** (mathematically guaranteed constraints).

## Decision Drivers
*   **Asset Agnosticism**: Must work with any ticker list provided by the user via `yfinance`.
*   **Regulatory Compliance**: Must strictly enforce diversity constraints (long-only, weights sum to 1, minimum weights).
*   **Latency**: Must return results within 50ms to ensure a "premium" UI feel.
*   **Explainability**: Must be compatible with SHAP for feature attribution.

## Considered Options

### Option 1: Retrain Multi-Head LSTM
*   **Pros**: Pure ML approach, potentially captures higher-order non-linearities.
*   **Cons**: Extremely high compute cost to retrain for every possible asset combination; remains a "Black Box."

### Option 2: Differentiable QP Solver (cvxpylayers)
*   **Pros**: End-to-end differentiable, mathematically elegant.
*   **Cons**: High complexity in setup; slower execution for small asset universes (~200ms+); overkill for v1.0.

### Option 3: Hybrid Universal MVO Solver (Scipy + Regime Gating)
*   **Pros**: Uses robust `scipy.optimize` for weight calculation; ML handles the "Regime Gating" (risk aversion); extremely fast (<30ms); guaranteed constraint satisfaction.
*   **Cons**: Decouples weight prediction from feature extraction.

## Decision
We decided to implement **Option 3: Hybrid Universal MVO Solver**.

## Rationale
The Hybrid approach provides the best balance of:
1.  **Flexibility**: By using `scipy.optimize.minimize` with the SLSQP method, we can optimize any number of assets in real-time.
2.  **Institutional Precision**: Constraints are enforced as hard mathematical boundaries rather than "learned" behaviors.
3.  **Performance**: Achieving sub-30ms latency satisfies the requirement for a premium, responsive dashboard experience.
4.  **Simplicity**: It leverages well-tested financial math (Mean-Variance Optimization) which is more easily audited by compliance teams.

## Consequences

### Positive
*   The system now supports any ticker universe input.
*   Weights are mathematically optimal and diversity-constrained (Min 2%).
*   The API latency dropped to ~25ms.
*   SHAP values now correctly attribute importance to the specific assets being optimized.

### Negative
*   We lost the "End-to-End" gradient flow from the solver back to the LSTM.
*   The solver relies on the quality of the covariance matrix calculated from `yfinance` data.

### Risks
*   **Data Quality**: Poor data from Yahoo Finance can lead to unstable optimization results ("Garbage In, Garbage Out").
*   **Mitigation**: Implemented input validation and NaN filtering in `api/main.py`.

## Implementation Notes
*   Solver method: `SLSQP`.
*   Risk Aversion: Dynamically scaled based on VIX regime.
*   Constraints: Diversity (lower bound 0.02) and Sum-to-One.
