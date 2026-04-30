---
phase: "05-backtest-and-interpretability"
wave: 1
---

# Plan: Backtest Engine

## Goal
Construct `backtest/engine.py` to evaluate true out-of-sample portfolio trajectories while harshly penalizing turnover using the 10bps transaction cost baseline.

## Dependencies
- `mlpo/config.py`

## Must Haves
- Correct vectorization of transaction penalties.
- Generates a dictionary of summary statistics (Sharpe, Volatility, Max Drawdown, Cumulative Return, Total Turnover).

## Task 1: Implement Simulator
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/config.py
</read_first>
<acceptance_criteria>
- `mlpo/backtest/engine.py` containing `BacktestEngine`.
- Accurately simulates the sequence $R_{net} = R_{gross} - Costs$.
</acceptance_criteria>
<action>
Create `mlpo/backtest/engine.py`. Build the `BacktestEngine` class. It receives `weights` (shape `[T, P]`) and `returns` (shape `[T, P]`). Calculate the absolute shift in weights period-over-period. Multiply the shift sum by `config.TRANSACTION_COST_BPS / 10000.0`. Subtract this from the gross portfolio return. Output standard metrics (annualized mean, annualized volatility, Sharpe, MDD) inside a `run()` method.
</action>

## Task 2: Build Backtest Tests
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/backtest/engine.py
</read_first>
<acceptance_criteria>
- `tests/test_backtest.py` proves penalty bounds are functionally locked.
</acceptance_criteria>
<action>
Create `tests/test_backtest.py`. Write unit tests ensuring `transaction_costs == 0.0` when $w_t == w_{t-1}$. Supply an exact mock vector of returns and weights to guarantee the output `cumulative_return` matches the deterministic manual calculation.
</action>
