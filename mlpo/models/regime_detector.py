"""
Module B — Real-Time Regime Detection.

Architecture:
    2-layer LSTM encoder (128 units) → Gaussian Mixture Model soft-gating.
    Computes dynamic risk budget β_t = Σ_k (γ_k · b_k).

    Tail-Risk Override:
        If VIX >= 35, the network output is hard-overridden to force
        >= 40% allocation to defensive assets (Fixed Income + Commodities).

Financial Intuition:
    In crisis regimes, learned risk budgets may be dangerously pro-cyclical.
    The VIX override acts as a circuit breaker — a deterministic safety net
    that overrides the stochastic regime detector during tail events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from mlpo.config import config


class RegimeDetector(nn.Module):
    """
    LSTM-GMM Regime Detection with hard VIX override.

    Parameters
    ----------
    n_macro : int
        Number of macro state variables (VIX, TED spread, etc.).
    hidden_size : int
        LSTM encoder hidden dimension.
    n_layers : int
        LSTM encoder depth.
    n_regimes : int
        Number of GMM components (typically 3: bull/neutral/bear).
    n_assets : int
        Portfolio size — determines risk budget vector dimension.
    """

    def __init__(
        self,
        n_macro: int = config.N_MACRO,
        hidden_size: int = config.REGIME_LSTM_HIDDEN,
        n_layers: int = config.REGIME_LSTM_LAYERS,
        n_regimes: int = config.N_REGIMES,
        n_assets: int = config.N_ASSETS,
    ) -> None:
        super().__init__()
        self.n_regimes = n_regimes
        self.n_assets = n_assets

        # ── LSTM Encoder ──────────────────────────
        self.lstm = nn.LSTM(
            input_size=n_macro,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=0.2 if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

        # ── GMM Soft-Gating ───────────────────────
        # γ_k: regime responsibilities (softmax over n_regimes)
        self.regime_logits = nn.Linear(hidden_size, n_regimes)

        # b_k: per-regime risk budget vectors
        # Each b_k is a learnable n_assets-dim vector (softmax-normalised)
        self.risk_budgets = nn.Parameter(
            torch.randn(n_regimes, n_assets) * 0.01
        )

    def forward(
        self,
        macro_features: torch.Tensor,
        vix_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic risk budget from macro state.

        Parameters
        ----------
        macro_features : torch.Tensor
            Shape ``[B, Seq, N_macro]`` — macro indicator time series.
        vix_values : torch.Tensor
            Shape ``[B]`` — current VIX level for tail-risk check.

        Returns
        -------
        beta : torch.Tensor
            Risk budget vector, shape ``[B, N_assets]``.
            Entries are non-negative and sum to 1.
        regime_probs : torch.Tensor
            Regime responsibilities γ_k, shape ``[B, N_regimes]``.
        """
        B = macro_features.shape[0]

        # ── LSTM encoding ─────────────────────────
        lstm_out, _ = self.lstm(macro_features)   # [B, Seq, H]
        h_last = lstm_out[:, -1, :]               # [B, H]
        h_last = self.layer_norm(h_last)

        # ── Regime probabilities (soft-gating) ────
        # γ_k via softmax ensures Σ γ_k = 1
        regime_probs = F.softmax(
            self.regime_logits(h_last), dim=-1
        )  # [B, K]

        # ── Per-regime risk budgets ───────────────
        # Softmax along asset dimension ensures each b_k sums to 1
        # Financial intuition: risk budget = fraction of total portfolio
        # risk each asset is allocated.
        budgets = F.softmax(self.risk_budgets, dim=-1)  # [K, P]

        # ── β_t = Σ_k (γ_k · b_k) ───────────────
        # Einstein summation: [B,K] × [K,P] → [B,P]
        beta = torch.einsum("bk,kp->bp", regime_probs, budgets)

        # ── VIX HARD OVERRIDE ─────────────────────
        # If VIX >= 35, force at least 40% to defensive assets.
        # This is a deterministic circuit breaker that cannot be
        # overridden by the learned network — critical for tail risk.
        vix_mask = (vix_values >= config.VIX_THRESHOLD)  # [B]

        if vix_mask.any():
            beta = self._apply_vix_override(beta, vix_mask)

        return beta, regime_probs

    def _apply_vix_override(
        self,
        beta: torch.Tensor,
        vix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Redistribute risk budget to force >= 40% defensive allocation.

        The override preserves differentiability for non-overridden
        samples in the batch; overridden samples get a detached budget.
        """
        beta_override = beta.clone()

        # Defensive indices: Fixed Income + Commodities
        defensive_idx = config.DEFENSIVE_INDICES
        equity_idx = list(range(len(config.EQUITIES)))

        for i in range(beta.shape[0]):
            if vix_mask[i]:
                current_defensive = beta[i, defensive_idx].sum()
                shortfall = config.DEFENSIVE_ALLOC - current_defensive

                if shortfall > 0:
                    # Reduce equity budgets proportionally
                    equity_total = beta[i, equity_idx].sum()
                    if equity_total > config.EPSILON:
                        reduction_ratio = shortfall / equity_total
                        reduction_ratio = min(reduction_ratio, 0.8)

                        beta_override[i, equity_idx] *= (1 - reduction_ratio)
                        # Redistribute to defensives equally
                        n_def = len(defensive_idx)
                        beta_override[i, defensive_idx] += shortfall / n_def

                # Re-normalise to sum=1
                beta_override[i] = beta_override[i] / (
                    beta_override[i].sum() + config.EPSILON
                )

        return beta_override
