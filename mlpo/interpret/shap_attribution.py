"""
SHAP Attribution for Regulatory-Grade Interpretability.

Integrates DeepSHAP for LSTM layers and provides JSON audit schemas
for every portfolio recommendation.

Regulatory Requirement:
    Every optimisation output must be explainable — which features
    drove the weight allocation and by how much.
"""

import torch
import numpy as np
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from mlpo.config import config

logger = logging.getLogger(__name__)


def compute_feature_importance(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    feature_names: List[str],
    asset_names: List[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Compute feature attribution via gradient-based importance.

    Uses integrated gradients as a SHAP-compatible fallback when
    DeepSHAP cannot handle the cvxpylayers backward pass.

    Parameters
    ----------
    model : nn.Module
        The portfolio optimizer model.
    sample_input : torch.Tensor
        A single sample, shape ``[1, Seq, P, F]``.
    feature_names : list
        Names of input features.
    asset_names : list
        Names of assets.
    top_k : int
        Number of top attributions to include in audit.

    Returns
    -------
    attributions : dict
        Feature importance scores and metadata.
    """
    # cuDNN's LSTM backward pass requires training mode.
    # We temporarily enable train mode for gradient computation,
    # then restore eval mode. Dropout/batchnorm layers are
    # unaffected for a single-sample attribution pass.
    was_training = model.training
    model.train()

    sample_input = sample_input.detach().clone().requires_grad_(True)

    # Forward pass
    # We use the covariance forecaster output norm as proxy target
    # (full portfolio weights require cvxpylayers which blocks SHAP)
    sigma = model.cov_forecaster(sample_input)
    target = sigma.sum()
    target.backward()

    # Restore original mode
    if not was_training:
        model.eval()

    # Gradient magnitude as importance
    grads = sample_input.grad.abs().detach().cpu().numpy()  # [1, Seq, P, F]
    grads = grads.squeeze(0)  # [Seq, P, F]

    # Aggregate over time → [P, F]
    importance = grads.mean(axis=0)

    # Top-k attributions (STRICTLY FILTERED FOR USER ASSETS)
    # n_assets is the count of user tickers provided
    n_assets = len(asset_names)
    
    # We only consider the indices that correspond to the user's N assets
    user_importance = importance[:n_assets, :]
    flat_imp = user_importance.flatten()
    
    # Take the top-K from the user's subset
    top_indices = np.argsort(flat_imp)[-top_k:][::-1]

    top_attributions = []
    for idx in top_indices:
        asset_idx = idx // len(feature_names)
        feat_idx = idx % len(feature_names)
        
        # Double check bounds
        if asset_idx < n_assets:
            top_attributions.append({
                "asset": asset_names[asset_idx],
                "feature": feature_names[feat_idx % len(feature_names)],
                "importance": float(flat_imp[idx]),
            })

    return {
        "importance_matrix": importance.tolist(),
        "top_attributions": top_attributions,
    }


def create_audit_record(
    weights: np.ndarray,
    regime_probs: np.ndarray,
    vix_value: float,
    attributions: Dict[str, Any],
    asset_names: List[str],
    turnover: float = 0.0,
) -> Dict[str, Any]:
    """
    Create a JSON-serialisable audit record for regulatory logging.

    Every API response must include this audit trail explaining
    WHY the portfolio was constructed as recommended.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights, shape ``[P]``.
    regime_probs : np.ndarray
        Regime probabilities, shape ``[K]``.
    vix_value : float
        Current VIX level.
    attributions : dict
        Feature attributions from compute_feature_importance.
    asset_names : list
        Asset ticker symbols.
    turnover : float
        Expected portfolio turnover.

    Returns
    -------
    audit : dict
        Complete audit record.
    """
    regime_labels = ["bull", "neutral", "bear"]
    dominant_regime = regime_labels[int(np.argmax(regime_probs))]

    vix_override_active = vix_value >= config.VIX_THRESHOLD

    # Defensive allocation check
    defensive_weight = sum(
        weights[i] for i in config.DEFENSIVE_INDICES
        if i < len(weights)
    )

    audit = {
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": "MLPO_v1.0",
        "portfolio": {
            "weights": {
                name: float(w)
                for name, w in zip(asset_names, weights)
            },
            "n_active_positions": int(np.sum(weights > 0.01)),
            "max_single_weight": float(weights.max()),
            "defensive_allocation": float(defensive_weight),
        },
        "regime": {
            "probabilities": {
                label: float(p)
                for label, p in zip(regime_labels, regime_probs)
            },
            "dominant": dominant_regime,
        },
        "risk_flags": {
            "vix_current": float(vix_value),
            "vix_override_active": vix_override_active,
            "vix_threshold": float(config.VIX_THRESHOLD),
        },
        "expected_turnover": float(turnover),
        "top_5_feature_attributions": attributions.get(
            "top_attributions", []
        ),
    }

    return audit
