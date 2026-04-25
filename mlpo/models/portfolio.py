"""
End-to-End Portfolio Optimization Module.

Assembles all sub-modules into a single differentiable nn.Module:

    Features → LSTM Covariance → Regime Detector → Risk Budgeting → Weights

Gradient flows from the composite financial loss (Sharpe + MDD + Turnover)
back through the convex solver and into the LSTM feature extractors.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict

from mlpo.config import config
from mlpo.models.lstm_covariance import LSTMCovarianceForecaster
from mlpo.models.regime_detector import RegimeDetector
from mlpo.models.attention import SparseAttention
from mlpo.models.risk_budgeting import DifferentiableRiskBudgeting


class PortfolioOptimizer(nn.Module):
    """
    End-to-end differentiable portfolio optimizer.

    Parameters
    ----------
    n_assets : int
        Number of assets.
    n_features : int
        Features per asset per timestep.
    n_macro : int
        Number of macro state variables.
    use_attention : bool
        Whether to apply sparse attention over LSTM embeddings.
    """

    def __init__(
        self,
        n_assets: int = config.N_ASSETS,
        n_features: int = config.N_FEATURES,
        n_macro: int = config.N_MACRO,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.use_attention = use_attention

        # ── Module A: Volatility Forecaster ───────
        self.cov_forecaster = LSTMCovarianceForecaster(
            n_assets=n_assets, n_features=n_features,
        )

        # ── Module B: Regime Detection ────────────
        self.regime_detector = RegimeDetector(
            n_macro=n_macro, n_assets=n_assets,
        )

        # ── Module D: Sparse Attention ────────────
        if use_attention:
            self.attention = SparseAttention(
                embed_dim=config.LSTM_HIDDEN,
                n_assets=n_assets,
            )
            # Projection from LSTM hidden to asset embeddings
            self.embed_proj = nn.Linear(
                config.LSTM_HIDDEN, config.LSTM_HIDDEN,
            )

        # ── Module C: Risk Budgeting ──────────────
        self.risk_budgeting = DifferentiableRiskBudgeting(
            n_assets=n_assets,
        )

    def forward(
        self,
        asset_features: torch.Tensor,
        macro_features: torch.Tensor,
        vix_values: torch.Tensor,
        w_prev: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: features → portfolio weights.

        Parameters
        ----------
        asset_features : torch.Tensor
            Shape ``[B, Seq, N_assets, N_features]``.
        macro_features : torch.Tensor
            Shape ``[B, Seq, N_macro]``.
        vix_values : torch.Tensor
            Shape ``[B]`` — current VIX level.
        w_prev : torch.Tensor, optional
            Previous weights, shape ``[B, N_assets]``.

        Returns
        -------
        result : dict
            'weights': [B, N_assets] — optimal portfolio weights.
            'sigma': [B, N_assets, N_assets] — predicted covariance.
            'beta': [B, N_assets] — risk budget.
            'regime_probs': [B, N_regimes] — regime responsibilities.
        """
        # ── Step 1: Forecast covariance matrix ────
        sigma = self.cov_forecaster(asset_features)  # [B, P, P]

        # ── Step 2: Detect regime → risk budget ───
        beta, regime_probs = self.regime_detector(
            macro_features, vix_values,
        )  # [B, P], [B, K]

        # ── Step 3: Solve for weights ─────────────
        # Clear intermediate gradients before the memory-intensive
        # cvxpylayers forward pass on RTX 3050.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        weights = self.risk_budgeting(sigma, beta, w_prev)  # [B, P]

        return {
            "weights": weights,
            "sigma": sigma,
            "beta": beta,
            "regime_probs": regime_probs,
        }

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
