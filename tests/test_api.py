import pytest
from fastapi.testclient import TestClient
from mlpo.api.main import app
from mlpo.config import config

@pytest.fixture(scope="module")
def client():
    # Need to manually call startup event for TestClient
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_predict_valid_payload(client):
    asset_returns = {ticker: 0.01 for ticker in config.ASSET_LIST}
    payload = {
        "asset_returns": asset_returns,
        "vix_value": 20.0
    }
    response = client.post("/v1/portfolio/optimize", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert "allocations" in data
    assert len(data["allocations"]) == config.N_ASSETS
    assert "regime" in data
    assert "risk_flags" in data
    assert data["risk_flags"]["vix_current"] == 20.0
    assert data["risk_flags"]["vix_override_active"] is False

def test_predict_invalid_payload(client):
    # Missing required vix_value
    asset_returns = {ticker: 0.01 for ticker in config.ASSET_LIST}
    payload = {
        "asset_returns": asset_returns
    }
    response = client.post("/v1/portfolio/optimize", json=payload)
    
    # Pydantic should block it
    assert response.status_code == 422
    
def test_predict_invalid_returns(client):
    # Exceeds the +/- 50% limit in the validator
    asset_returns = {ticker: 0.01 for ticker in config.ASSET_LIST}
    asset_returns[config.ASSET_LIST[0]] = 0.6  # 60%
    payload = {
        "asset_returns": asset_returns,
        "vix_value": 20.0
    }
    response = client.post("/v1/portfolio/optimize", json=payload)
    
    assert response.status_code == 422
