# Phase 6 Research: FastAPI Layer

## 1. FastAPI Architecture (`api/main.py`)
To expose the MLPO v1.0 engine to downstream execution algorithms or front-end dashboards, we need a high-performance HTTP layer.
- **FastAPI**: Chosen for its async capabilities and native validation via Pydantic. It automatically generates OpenAPI documentation which is crucial for institutional handoffs.
- **Dependency Injection**: We need a way to load the `PortfolioOptimizer` model and its trained weights (if available) into memory *once* upon server startup, and pass it cleanly into the routing functions.

## 2. Request / Response Contracts (`api/schemas.py`)
- **Input**: The endpoint needs to receive market data. Given the model's dimensions, it expects `[Seq, Assets, Features]` for historical data, plus macro features. However, sending raw multi-dimensional tensors via JSON is clunky. We will expect a structured JSON payload defining recent daily rows, which the API will internally convert into the `torch.Tensor` structures required by `mlpo.models.portfolio`.
- **Output**: The output must rigidly match the institutional compliance format built in Phase 5:
    - Target `weights` mapped to asset tickers.
    - `regime` dominant state.
    - `vix_override_active` boolean flag.
    - `top_5_feature_attributions` for SHAP compliance.

## 3. Server Safety & Error Handling
- PyTorch execution should occur inside `torch.no_grad()` to prevent memory leaks during inference.
- Error codes must be explicit. `400 Bad Request` if data dimensions don't match `config.N_ASSETS`. `500 Internal Server Error` if matrix inversion fails inside `cvxpylayers`.
