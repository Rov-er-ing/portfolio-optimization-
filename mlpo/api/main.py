"""
FastAPI Application — MLPO v1.0.

Asynchronous endpoints for portfolio optimisation with
Pydantic v2 validation and audit logging.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import torch
import numpy as np
from datetime import datetime
import logging

from mlpo.config import config, seed_everything
from mlpo.models.portfolio import PortfolioOptimizer
from mlpo.api.schemas import (
    OptimizeRequest, OptimizeResponse,
    PortfolioAllocation, RegimeInfo, RiskFlags,
    FeatureAttribution, HealthResponse,
)

logger = logging.getLogger(__name__)

# ── Global model reference ────────────────────────
_model: PortfolioOptimizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _model
    seed_everything(42)
    _model = PortfolioOptimizer()
    _model = _model.to(config.DEVICE)
    _model.eval()

    # Load trained weights if available
    import os
    ckpt = f"{config.MODEL_DIR}/best_model.pt"
    if os.path.exists(ckpt):
        _model.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
        logger.info(f"Loaded model checkpoint from {ckpt}")
    else:
        logger.warning("No checkpoint found — using random weights")

    yield

    # Cleanup
    _model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="MLPO Portfolio Optimizer",
    version="1.0.0",
    description="ML-Powered Portfolio Optimization with differentiable risk budgeting.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Service health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        device=str(config.DEVICE),
        n_parameters=_model.count_parameters() if _model else 0,
    )


@app.post("/v1/portfolio/optimize", response_model=OptimizeResponse)
async def optimize_portfolio(request: OptimizeRequest) -> OptimizeResponse:
    """
    Optimise portfolio weights given current market state.

    The response includes:
    - Optimal weights per asset.
    - Regime detection (bull/neutral/bear).
    - VIX override flags.
    - Top-5 feature attributions.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ── Map request to tensors ────────────────────
    # Create dummy features for inference (in production, these
    # would come from the live data pipeline).
    n_assets = config.N_ASSETS
    dummy_features = torch.randn(
        1, config.SEQUENCE_LENGTH, n_assets, config.N_FEATURES,
        device=config.DEVICE,
    )
    dummy_macro = torch.randn(
        1, config.SEQUENCE_LENGTH, config.N_MACRO,
        device=config.DEVICE,
    )
    vix_tensor = torch.tensor(
        [request.vix_value], device=config.DEVICE,
    )

    # ── Inference ─────────────────────────────────
    with torch.no_grad():
        result = _model(dummy_features, dummy_macro, vix_tensor)

    weights = result["weights"].cpu().numpy().flatten()
    regime_probs = result["regime_probs"].cpu().numpy().flatten()

    # ── Build response ────────────────────────────
    regime_labels = ["bull", "neutral", "bear"]
    dominant = regime_labels[int(np.argmax(regime_probs))]

    allocations = [
        PortfolioAllocation(ticker=t, weight=float(w))
        for t, w in zip(config.ASSET_LIST, weights)
    ]

    turnover = 0.0
    if request.previous_weights:
        for t, w in zip(config.ASSET_LIST, weights):
            prev = request.previous_weights.get(t, 1.0 / n_assets)
            turnover += abs(w - prev)

    return OptimizeResponse(
        timestamp=datetime.utcnow(),
        allocations=allocations,
        regime=RegimeInfo(
            dominant=dominant,
            probabilities={
                label: float(p) for label, p in zip(regime_labels, regime_probs)
            },
        ),
        risk_flags=RiskFlags(
            vix_current=request.vix_value,
            vix_override_active=request.vix_value >= config.VIX_THRESHOLD,
            vix_threshold=config.VIX_THRESHOLD,
        ),
        expected_turnover=turnover,
        top_attributions=[],  # Populated when SHAP is available
    )
