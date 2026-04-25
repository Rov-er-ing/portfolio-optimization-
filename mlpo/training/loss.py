"""
Composite Financial Loss Function.

L(θ) = −Sharpe + γ₁·MDD + γ₂·Turnover + γ₃·||θ||²

Financial Intuition:
    Standard MSE is agnostic to financial objectives.  This loss directly
    optimises for risk-adjusted returns (Sharpe), penalises tail risk
    (MaxDrawdown), and controls implementation costs (Turnover).

    The negative Sharpe term is the primary driver; the penalties prevent
    the optimiser from chasing returns at the expense of tail risk and
    trading costs.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from mlpo.config import config


class CompositeFinancialLoss(nn.Module):
    """
    Multi-objective portfolio loss.

    Parameters
    ----------
    gamma_mdd : float
        Max drawdown penalty weight.
    gamma_turnover : float
        Turnover penalty weight.
    gamma_l2 : float
        Parameter L2 regularisation weight.
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation.
    """

    def __init__(
        self,
        gamma_mdd: float = config.LOSS_GAMMA_MDD,
        gamma_turnover: float = config.LOSS_GAMMA_TURNOVER,
        gamma_l2: float = config.LOSS_GAMMA_L2,
        risk_free_rate: float = config.RISK_FREE_RATE,
    ) -> None:
        super().__init__()
        self.gamma_mdd = gamma_mdd
        self.gamma_turnover = gamma_turnover
        self.gamma_l2 = gamma_l2
        # Daily risk-free rate from annualised
        self.rf_daily = risk_free_rate / config.TRADING_DAYS

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        w_prev: Optional[torch.Tensor] = None,
        model_params: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.

        Parameters
        ----------
        weights : torch.Tensor
            Portfolio weights, shape ``[B, P]``.
        returns : torch.Tensor
            Asset returns at time t+1, shape ``[B, P]``.
        w_prev : torch.Tensor, optional
            Previous weights for turnover, shape ``[B, P]``.
        model_params : list, optional
            Model parameters for L2 regularisation.

        Returns
        -------
        losses : dict
            'total': scalar loss for backward().
            'sharpe': Sharpe component.
            'mdd': Max drawdown component.
            'turnover': Turnover component.
            'l2': L2 component.
        """
        # ── Portfolio returns ─────────────────────
        # R_p = Σ w_i · r_i (weighted portfolio return per sample)
        port_returns = (weights * returns).sum(dim=-1)  # [B]

        # ── Sharpe Ratio (differentiable) ─────────
        # Annualise using √252: Sharpe = E[R_p - Rf] / σ(R_p) × √252
        excess = port_returns - self.rf_daily
        sharpe = excess.mean() / (excess.std() + config.EPSILON)
        sharpe = sharpe * (config.TRADING_DAYS ** 0.5)
        neg_sharpe = -sharpe

        # ── Max Drawdown (batch approximation) ────
        # For a single batch, we compute the drawdown from the cumulative
        # wealth curve within the batch.
        cum_returns = (1 + port_returns).cumprod(dim=0)
        running_max = cum_returns.cummax(dim=0).values
        drawdowns = (running_max - cum_returns) / (running_max + config.EPSILON)
        mdd = drawdowns.max()

        # ── Turnover ──────────────────────────────
        if w_prev is not None:
            turnover = (weights - w_prev).abs().sum(dim=-1).mean()
        else:
            turnover = torch.tensor(0.0, device=weights.device)

        # ── L2 Regularisation ─────────────────────
        l2_loss = torch.tensor(0.0, device=weights.device)
        if model_params is not None:
            for p in model_params:
                l2_loss = l2_loss + p.pow(2).sum()

        # ── Composite ─────────────────────────────
        total = (
            neg_sharpe
            + self.gamma_mdd * mdd
            + self.gamma_turnover * turnover
            + self.gamma_l2 * l2_loss
        )

        return {
            "total": total,
            "sharpe": -neg_sharpe,  # Report positive Sharpe
            "mdd": mdd,
            "turnover": turnover,
            "l2": l2_loss,
        }
