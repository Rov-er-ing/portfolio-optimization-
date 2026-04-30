import torch
import numpy as np
from mlpo.interpret.shap_attribution import compute_feature_importance, create_audit_record
from mlpo.models.portfolio import PortfolioOptimizer
from mlpo.config import config

def test_feature_importance_dimensions():
    """
    Test that the gradient-based attribution generator returns scores matching
    the exact [Assets, Features] matrix limits without breaking the graph.
    """
    model = PortfolioOptimizer(
        n_assets=config.N_ASSETS,
        n_features=config.N_FEATURES,
        n_macro=config.N_MACRO
    )
    
    # Dummy inputs
    seq_len = 10
    sample_input = torch.randn(1, seq_len, config.N_ASSETS, config.N_FEATURES)
    feature_names = ["z_score", "volatility"]
    asset_names = config.ASSET_LIST
    
    attributions = compute_feature_importance(
        model=model,
        sample_input=sample_input,
        feature_names=feature_names,
        asset_names=asset_names,
        top_k=5
    )
    
    # Check dictionary outputs
    assert "importance_matrix" in attributions
    assert "top_attributions" in attributions
    
    imp_matrix = np.array(attributions["importance_matrix"])
    assert imp_matrix.shape == (config.N_ASSETS, config.N_FEATURES), f"Expected shape {(config.N_ASSETS, config.N_FEATURES)}, got {imp_matrix.shape}"
    assert len(attributions["top_attributions"]) == 5, "Expected top_k=5 attributions"

def test_audit_record_logging():
    """
    Test the dictionary structure of the regulatory audit record, focusing on
    VIX overrides.
    """
    weights = np.array([1.0 / config.N_ASSETS] * config.N_ASSETS)
    regime_probs = np.array([0.1, 0.2, 0.7]) # Bear
    vix_value = 40.0 # Above threshold
    attributions = {"top_attributions": []}
    asset_names = config.ASSET_LIST
    
    audit = create_audit_record(
        weights=weights,
        regime_probs=regime_probs,
        vix_value=vix_value,
        attributions=attributions,
        asset_names=asset_names
    )
    
    assert audit["risk_flags"]["vix_override_active"] is True, "VIX override flag should be True"
    assert audit["regime"]["dominant"] == "bear", "Dominant regime should be bear"
    assert "portfolio" in audit
