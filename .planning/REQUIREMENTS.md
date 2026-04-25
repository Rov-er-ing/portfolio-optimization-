# REQUIREMENTS: MLPO v1.0

## 1. Functional Requirements

### 1.1 Data Ingestion & Engineering
- **REQ-1.1.1:** Fetch equity, fixed income, and commodity data (45 instruments) via yfinance or Bloomberg.
- **REQ-1.1.2:** Implement lookahead-free feature engineering with rolling statistics.
- **REQ-1.1.3:** Lazy loading of data windows to conserve RAM.

### 1.2 Forecasting Modules
- **REQ-1.2.1:** LSTM architecture for conditional covariance forecasting.
- **REQ-1.2.2:** Output Cholesky factors to guarantee PSD covariance matrices.
- **REQ-1.2.3:** Regime detection via GMM + LSTM with VIX-based hard constraints.

### 1.3 Optimization Layer
- **REQ-1.3.1:** Integrate `cvxpylayers` for end-to-end differentiability.
- **REQ-1.3.2:** Implement Sparse Attention mechanism for scalable asset interactions.

### 1.4 Training & Backtesting
- **REQ-1.4.1:** Composite financial loss function (Sharpe + Drawdown + Turnover).
- **REQ-1.4.2:** Walk-forward simulation engine with transaction cost modeling.

### 1.5 Interpretability & API
- **REQ-1.5.1:** SHAP attribution for both LSTM and Optimization layers.
- **REQ-1.5.2:** FastAPI endpoints with Pydantic v2 validation and audit logging.

## 2. Technical Constraints
- **CON-2.1:** Maximum VRAM usage of 6GB (AMP FP16 required).
- **CON-2.2:** Maximum RAM usage of 16GB.
- **CON-2.3:** Python 3.11 compatibility (configured in .env).

## 3. Success Metrics
- **MET-3.1:** Validated gradient flow from loss to feature layers.
- **MET-3.2:** Backtest Sharpe Ratio improvement vs benchmark risk parity.
- **MET-3.3:** Execution time for /optimize endpoint < 500ms.
