# Phase 6 Validation: API Layer

## Acceptance Criteria
1. **API Endpoints**:
   - `GET /health` returns `{"status": "ok"}`.
   - `POST /predict` accepts a JSON payload of market data and returns the exact dictionary format matching `create_audit_record()`.
2. **Data Validation**:
   - Pydantic models automatically reject payloads missing required fields or possessing incorrect asset dimensions, returning `422 Unprocessable Entity`.
3. **Inference Safety**:
   - The model must execute in evaluation mode (`model.eval()`) without consuming gradient memory.

## Test Strategy (Pytest & FastAPI TestClient)

### 1. Endpoint Testing (`tests/test_api.py`)
- Use `fastapi.testclient.TestClient` to simulate HTTP requests.
- **Health Check**: Assert HTTP 200 on `/health`.
- **Valid Inference**: Post a correctly sized mock dictionary mimicking the API schema. Assert HTTP 200 and verify that the response JSON contains keys: `portfolio`, `regime`, `risk_flags`, and `top_5_feature_attributions`.
- **Invalid Inference**: Post a dictionary with missing macro features or an incorrect number of assets. Assert HTTP 422.

## Manual-Only Verifications
| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Server Boot | Startup | Confirm Uvicorn boots | Run `uvicorn mlpo.api.main:app` and verify it binds to `0.0.0.0:8000` without crashing. |
