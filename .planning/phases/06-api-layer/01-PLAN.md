---
phase: "06-api-layer"
wave: 1
---

# Plan: API Routes and Schemas

## Goal
Build `api/schemas.py` for Pydantic validation and `api/main.py` utilizing FastAPI to serve the portfolio optimization engine.

## Dependencies
- `mlpo/models/portfolio.py`
- `mlpo/interpret/shap_attribution.py`
- `fastapi` and `pydantic`

## Must Haves
- `POST /predict` endpoint wrapping the PortfolioOptimizer.
- Memory-safe inference using `torch.no_grad()` (except when capturing gradients for Interpretability).

## Task 1: Create API Schemas
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/config.py
</read_first>
<acceptance_criteria>
- `mlpo/api/schemas.py` defines `MarketDataInput` and `OptimizationResponse`.
</acceptance_criteria>
<action>
Create `mlpo/api/schemas.py`. Write Pydantic `BaseModel` classes. `MarketDataInput` should expect `asset_features` (list of lists) and `macro_features`. `OptimizationResponse` should validate the output structure matching the JSON from `create_audit_record()`.
</action>

## Task 2: Create FastAPI Application
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/api/schemas.py
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/models/portfolio.py
</read_first>
<acceptance_criteria>
- `mlpo/api/main.py` runs a FastAPI app with `/health` and `/predict`.
</acceptance_criteria>
<action>
Create `mlpo/api/main.py`. Instantiate `FastAPI()`. Initialize a global `PortfolioOptimizer` model. Write `GET /health` returning `{"status": "ok", "model": "MLPO_v1.0"}`. Write `POST /predict` taking `MarketDataInput`. In the handler, convert the JSON lists into PyTorch tensors, run the model to get weights and regimes, trigger `compute_feature_importance` and `create_audit_record`, and return the audit JSON as the HTTP response.
</action>
