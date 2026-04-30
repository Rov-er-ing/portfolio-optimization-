---
phase: "04-loss-and-trainer"
wave: 1
---

# Plan: Composite Financial Loss

## Goal
Implement the `CompositeLoss` class (`training/loss.py`) prioritizing Differentiable Sharpe Ratio, Differentiable Maximum Drawdown, and Turnover penalties.

## Dependencies
- `config.py`

## Must Haves
- Uses `torch` exclusively for math constraints to preserve the Autograd graph.
- Handles division-by-zero vulnerabilities in Sharpe ratios via `config.EPSILON`.
- Test suite verifying gradient preservation.

## Task 1: Build CompositeLoss
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/config.py
</read_first>
<acceptance_criteria>
- `mlpo/training/loss.py` contains `CompositeLoss(nn.Module)`.
- Loss function implements standard mathematical approximations of MDD, Sharpe, and Turnover.
</acceptance_criteria>
<action>
Create `mlpo/training/loss.py`. Add `CompositeLoss` inheriting `nn.Module`. Implement the `forward(portfolio_returns, weights, prev_weights)` method. Calculate Sharpe as `mean(returns) / (std(returns) + epsilon)`. Calculate Turnover as `sum(abs(weights - prev_weights))`. Compute an approximate differentiable max drawdown. Combine them using the `LOSS_GAMMA_` coefficients defined in `config.py`.
</action>

## Task 2: Build Loss Tests
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/training/loss.py
</read_first>
<acceptance_criteria>
- `tests/test_loss.py` verifies output scalar format and backprop gradients.
</acceptance_criteria>
<action>
Create `tests/test_loss.py`. Build `test_composite_loss_gradients()`. Initialize dummy return sequences and weights with `requires_grad=True`. Call `CompositeLoss()`, verify the output is a standard `0D` scalar, and check that gradients flow backward securely without generating `NaN` values.
</action>
