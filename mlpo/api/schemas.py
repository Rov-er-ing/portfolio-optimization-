"""
Pydantic v2 I/O Validation Schemas.

Strict typing for all API request/response payloads.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from datetime import datetime


class OptimizeRequest(BaseModel):
    """Request schema for /v1/portfolio/optimize."""

    asset_returns: Dict[str, float] = Field(
        ...,
        description="Current period returns per asset (ticker → return).",
        min_length=1,
    )
    vix_value: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current VIX level.",
    )
    previous_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Previous portfolio weights for turnover calculation.",
    )

    @field_validator("asset_returns")
    @classmethod
    def validate_returns(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure returns are within plausible daily range."""
        for ticker, ret in v.items():
            if abs(ret) > 0.5:  # ±50% daily move is implausible
                raise ValueError(
                    f"Return for {ticker} = {ret:.4f} exceeds ±50% daily limit"
                )
        return v


class PortfolioAllocation(BaseModel):
    """Individual asset allocation."""

    ticker: str
    weight: float = Field(ge=0.0, le=1.0)
    risk_contribution: Optional[float] = None


class RegimeInfo(BaseModel):
    """Regime detection output."""

    dominant: str = Field(description="bull / neutral / bear")
    probabilities: Dict[str, float]


class RiskFlags(BaseModel):
    """Risk override status."""

    vix_current: float
    vix_override_active: bool
    vix_threshold: float


class FeatureAttribution(BaseModel):
    """Single feature importance entry."""

    asset: str
    feature: str
    importance: float


class OptimizeResponse(BaseModel):
    """Response schema for /v1/portfolio/optimize."""

    timestamp: datetime
    model_version: str = "MLPO_v1.0"
    allocations: List[PortfolioAllocation]
    efficiency_score: float
    regime: RegimeInfo
    risk_flags: RiskFlags
    expected_turnover: float
    top_attributions: List[FeatureAttribution]
    sharpe_estimate: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    device: str
    n_parameters: int
