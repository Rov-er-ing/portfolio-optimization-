# ROADMAP: MLPO v1.0

## Milestone 1: Core Engine & Data Pipeline
*Goal: Establish the differentiable foundation and memory-efficient data flow.*

### Phase 1: Environment & Foundation
- [ ] **Task 1.1:** Setup project structure and `config.py` (hyperparameters).
- [ ] **Task 1.2:** Implement `data/ingestion.py` for multi-asset fetching.
- [ ] **Task 1.3:** Develop `data/features.py` with lookahead-free rolling stats.
- [ ] **Task 1.4:** Create `data/dataset.py` with lazy-loading and windowing logic.

### Phase 2: Forecasting & Regime Detection
- [ ] **Task 2.1:** Implement `models/lstm_covariance.py` (Cholesky factorization).
- [ ] **Task 2.2:** Build `models/regime_detector.py` (GMM + VIX Override).
- [ ] **Task 2.3:** Unit tests for tensor shapes and PSD validity.

## Milestone 2: Optimization & Training
*Goal: Integrate convex layers and train for financial outcomes.*

### Phase 3: Differentiable Optimization
- [ ] **Task 3.1:** Implement `models/risk_budgeting.py` using `cvxpylayers`.
- [ ] **Task 3.2:** Develop `models/attention.py` (Sparse linear attention).
- [ ] **Task 3.3:** Gradient flow verification tests.

### Phase 4: Loss & Trainer
- [ ] **Task 4.1:** Implement `training/loss.py` (Composite Financial Loss).
- [ ] **Task 4.2:** Build `training/trainer.py` (AMP integration + Walk-forward loop).
- [ ] **Task 4.3:** Integrate `training/hyperopt.py` with Optuna.

## Milestone 3: Backtesting & Deployment
*Goal: Validate performance and expose via API.*

### Phase 5: Backtest & Interpretability
- [ ] **Task 5.1:** Build `backtest/engine.py` (Simulation + Transaction costs).
- [ ] **Task 5.2:** Implement `interpret/shap_attribution.py` (Audit logging).

### Phase 6: API Layer
- [ ] **Task 6.1:** Create `api/main.py` (FastAPI) and Pydantic schemas.
- [ ] **Task 6.2:** Final E2E verification and documentation.
