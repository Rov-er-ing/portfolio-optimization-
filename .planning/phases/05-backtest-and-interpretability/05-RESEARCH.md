# Phase 5 Research: Backtest Engine & Interpretability

## 1. Backtest Simulation (`backtest/engine.py`)
While `trainer.py` handles the mathematical compilation of weights and Sharpe ratios, the **Backtest Engine** measures out-of-sample practical execution.

- **Returns Calculation**: The core sequence resolves to $R_p = \sum_{i=1}^P w_i r_i$.
- **Transaction Costs**: Financial models often fail because they trade too heavily. Re-balancing costs must be strictly enforced. The PRD and `config.py` define a 10 basis point penalty: `TRANSACTION_COST_BPS = 10.0`.
- **Cost Formula**: $\text{Cost}_t = \sum_{i=1}^P |w_{t, i} - w_{t-1, i}| \times (\text{bps} / 10000)$. The gross portfolio return is then penalized by this cost.
- **Reporting Metrics**: The backtester generates final cumulative return trajectories, annualized volatility, Sortino ratio (downside risk), and maximum drawdown to confirm validation parameters.

## 2. SHAP Attribution (`interpret/shap_attribution.py`)
Neural networks applied to finance frequently encounter the "Black Box" problem, prohibiting their use by Institutional Risk Committees.
- **Integrated Gradients (Captum)**: Due to the complexity of the differentiable solver, direct SHAP execution can break on the convex layer. Instead, PyTorch's `captum` library, specifically **Integrated Gradients** or **DeepLift**, is used to map attribution scores backwards from the final output weights through the Sparse Attention maps and onto the raw macroeconomic inputs (e.g., how much did `TEDrate` influence the equity allocation).
- **VIX Override Tracing**: The regime override is deterministic. The interpretability module must log exactly when the VIX threshold broke through `35.0` to force defensive asset allocation.
