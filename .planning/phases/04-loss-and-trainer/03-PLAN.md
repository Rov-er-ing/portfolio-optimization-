---
phase: "04-loss-and-trainer"
wave: 3
---

# Plan: Hyperparameter Optimization

## Goal
Integrate Optuna (`training/hyperopt.py`) with `MedianPruner` support to search for optimal neural architecture layers and optimization parameters.

## Dependencies
- `mlpo/training/trainer.py`

## Must Haves
- Force stops trials via `MedianPruner` after 20 epochs to conserve GPU availability.

## Task 1: Optuna Objective and Runner
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/config.py
</read_first>
<acceptance_criteria>
- `mlpo/training/hyperopt.py` contains `run_optimization()` and `objective()`.
- Optuna MedianPruner is initialized inside the study.
</acceptance_criteria>
<action>
Create `mlpo/training/hyperopt.py`. Configure an `optuna.create_study` with `direction="maximize"` (we are maximizing Sharpe ultimately, or minimizing negative Sharpe) and `pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)`. Write the `objective(trial)` function simulating search space variables (like learning rates, LSTM dropout, or regularization coefficients), which invokes the `PortfolioTrainer` and returns validation performance.
</action>
