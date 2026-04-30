---
phase: "06-api-layer"
wave: 2
---

# Plan: API Testing and E2E

## Goal
Establish `tests/test_api.py` validating that the FastAPI HTTP bindings successfully receive multidimensional JSON sequences, convert them accurately, trigger the neural solver, and return the mandated compliance JSON payload.

## Dependencies
- `mlpo/api/main.py`
- `fastapi.testclient`

## Must Haves
- Full testing of the `POST /predict` endpoint using mock payload dictionaries.
- Verification of 422 HTTP responses when provided malformed shapes (preventing tensor crashes on the solver side).

## Task 1: Build API Endpoint Tests
<read_first>
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/api/main.py
- c:/Users/Shiti/OneDrive/ドキュメント/portfolio/mlpo/config.py
</read_first>
<acceptance_criteria>
- `tests/test_api.py` is executable by Pytest and achieves 100% test coverage over `main.py`.
</acceptance_criteria>
<action>
Create `tests/test_api.py`. Instantiate `TestClient(app)` from FastAPI. Write `test_health_check()` asserting standard 200 OK. Write `test_predict_valid_payload()` sending `asset_features` mapped to exactly `config.N_ASSETS` arrays and assert that the response JSON dictionary exactly conforms to the regulatory `audit` schema built in Phase 5. Write `test_predict_invalid_payload()` sending missing keys, asserting HTTP `422 Unprocessable Entity` is safely triggered by Pydantic before any PyTorch logic fires.
</action>
