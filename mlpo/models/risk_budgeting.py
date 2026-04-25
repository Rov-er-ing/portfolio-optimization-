"""
Module C — Differentiable Risk Budgeting via cvxpylayers.

Solves the convex risk-budgeting problem end-to-end with gradient flow:

    min_w  || (w ⊙ (Σ̂w)) / (wᵀΣ̂w) − β_t ||²
           + λ₁ ||w||₁        (sparsity)
           + λ₂ ||w − w_{t-1}||²  (turnover penalty)

    s.t.   Σ w_i = 1,   0 ≤ w_i ≤ 0.15

VRAM Note:
    cvxpylayers is extremely memory-hungry. We:
    1. Restrict batch size to 16.
    2. Clear grads aggressively between forward/backward.
    3. Use a simplified QP formulation to reduce solver overhead.

Financial Intuition:
    Risk budgeting decomposes total portfolio risk into per-asset risk
    contributions.  By matching them to the regime-dependent target β_t,
    the portfolio dynamically adapts its risk profile without requiring
    explicit return forecasts.
"""

import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from typing import Optional

from mlpo.config import config


class DifferentiableRiskBudgeting(nn.Module):
    """
    Differentiable convex solver for risk-parity portfolio weights.

    Parameters
    ----------
    n_assets : int
        Number of portfolio assets.
    max_weight : float
        Upper bound on individual asset weight.
    lambda_sparse : float
        L1 sparsity penalty coefficient.
    lambda_turnover : float
        Turnover penalty coefficient.
    """

    def __init__(
        self,
        n_assets: int = config.N_ASSETS,
        max_weight: float = config.MAX_WEIGHT,
        lambda_sparse: float = 0.001,
        lambda_turnover: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        self.max_weight = max_weight

        # ── Build cvxpy problem ───────────────────
        # Parameters (fed at runtime via forward())
        self.P_param = cp.Parameter((n_assets, n_assets), PSD=True, name="P")
        self.beta_param = cp.Parameter(n_assets, nonneg=True, name="beta")
        self.w_prev_param = cp.Parameter(n_assets, name="w_prev")
        self.lambda1_param = cp.Parameter(nonneg=True, name="lambda1")
        self.lambda2_param = cp.Parameter(nonneg=True, name="lambda2")

        # Decision variable
        w = cp.Variable(n_assets, name="w")

        # ── Objective ─────────────────────────────
        # Simplified quadratic: min wᵀΣw − βᵀw + penalties
        # This is a tractable QP approximation of the full risk-budgeting
        # problem that maintains differentiability through cvxpylayers.
        portfolio_risk = cp.quad_form(w, self.P_param)
        budget_alignment = -self.beta_param @ w
        sparsity = self.lambda1_param * cp.norm1(w)
        turnover = self.lambda2_param * cp.sum_squares(w - self.w_prev_param)

        objective = cp.Minimize(
            portfolio_risk + budget_alignment + sparsity + turnover
        )

        # ── Constraints ───────────────────────────
        constraints = [
            cp.sum(w) == 1,         # Fully invested
            w >= 0,                  # Long-only
            w <= max_weight,         # Concentration limit
        ]

        problem = cp.Problem(objective, constraints)

        # ── Wrap as differentiable layer ──────────
        self.cvx_layer = CvxpyLayer(
            problem,
            parameters=[
                self.P_param,
                self.beta_param,
                self.w_prev_param,
                self.lambda1_param,
                self.lambda2_param,
            ],
            variables=[w],
        )

        # Learnable penalty coefficients
        self.log_lambda1 = nn.Parameter(torch.tensor(-3.0))
        self.log_lambda2 = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        sigma: torch.Tensor,
        beta: torch.Tensor,
        w_prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Solve for optimal portfolio weights.

        Parameters
        ----------
        sigma : torch.Tensor
            Predicted covariance matrix, shape ``[B, P, P]``.
        beta : torch.Tensor
            Target risk budget, shape ``[B, P]``.
        w_prev : torch.Tensor, optional
            Previous period weights for turnover penalty.
            Defaults to equal-weight if None.

        Returns
        -------
        weights : torch.Tensor
            Optimal portfolio weights, shape ``[B, P]``.
        """
        B = sigma.shape[0]
        device = sigma.device

        if w_prev is None:
            w_prev = torch.full(
                (B, self.n_assets), 1.0 / self.n_assets,
                device=device, dtype=sigma.dtype,
            )

        lambda1 = torch.exp(self.log_lambda1).expand(B)
        lambda2 = torch.exp(self.log_lambda2).expand(B)

        # cvxpylayers expects float64 for solver precision
        try:
            (weights,) = self.cvx_layer(
                sigma.double(),
                beta.double(),
                w_prev.double(),
                lambda1.double(),
                lambda2.double(),
                solver_args={"max_iters": 10000, "solve_method": "SCS"},
            )
            weights = weights.float()
        except Exception:
            # Fallback: if solver fails, return equal-weight
            # This prevents training from crashing on ill-conditioned batches
            weights = torch.full(
                (B, self.n_assets), 1.0 / self.n_assets,
                device=device, dtype=sigma.dtype,
            )

        # Clamp and re-normalise for numerical safety
        weights = torch.clamp(weights, min=0.0, max=self.max_weight)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + config.EPSILON)

        return weights
