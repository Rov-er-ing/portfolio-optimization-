"""
Module A — LSTM Volatility Forecaster.

Predicts the lower-triangular Cholesky factor L of the conditional
covariance matrix, enforcing positive semi-definiteness by construction:

    Σ̂(t) = L · Lᵀ + ε·I

Architecture: 3-layer LSTM → Linear → reshape to lower-triangular L.

Memory Budget (RTX 3050, 6 GB VRAM):
    - Batch=16, Seq=60, Assets=45, Hidden=256
    - Forward pass ≈ 1.2 GB;  Backward ≈ 2.4 GB;  Optimizer ≈ 0.8 GB
    - Total ≈ 4.4 GB → safe with AMP headroom.
"""

import torch
import torch.nn as nn
from typing import Tuple

from mlpo.config import config


class LSTMCovarianceForecaster(nn.Module):
    """
    Forecasts the conditional covariance matrix Σ̂(t) via Cholesky factorisation.

    Parameters
    ----------
    n_assets : int
        Number of assets in the universe.
    n_features : int
        Number of input features per asset per timestep.
    hidden_size : int
        LSTM hidden dimension.
    n_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Inter-layer dropout rate.

    Notes
    -----
    The network predicts a vector of length n_assets*(n_assets+1)/2
    which is reshaped into a lower-triangular matrix L.  The diagonal
    of L is exponentiated to guarantee strict positivity, which in turn
    guarantees that Σ = L·Lᵀ + ε·I is positive definite.
    """

    def __init__(
        self,
        n_assets: int = config.N_ASSETS,
        n_features: int = config.N_FEATURES,
        hidden_size: int = config.LSTM_HIDDEN,
        n_layers: int = config.LSTM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.n_assets = n_assets
        # Number of free parameters in a lower-triangular matrix
        self.cholesky_size = n_assets * (n_assets + 1) // 2

        # ── LSTM encoder ──────────────────────────
        # Input: [Batch, Seq, n_assets * n_features]
        self.lstm = nn.LSTM(
            input_size=n_assets * n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        # Layer normalisation after LSTM — stabilises gradients on consumer GPU
        self.layer_norm = nn.LayerNorm(hidden_size)

        # ── Projection to Cholesky vector ─────────
        self.fc = nn.Linear(hidden_size, self.cholesky_size)

        # Pre-compute lower-triangular indices (register as buffer → moves with .to(device))
        tril_indices = torch.tril_indices(n_assets, n_assets)
        self.register_buffer("tril_row", tril_indices[0])
        self.register_buffer("tril_col", tril_indices[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: features → covariance matrix.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``[B, Seq, N_assets, N_features]``.

        Returns
        -------
        sigma : torch.Tensor
            Predicted covariance matrix, shape ``[B, N_assets, N_assets]``.
            Guaranteed PSD by Cholesky construction.
        """
        B, S, P, F = x.shape
        assert P == self.n_assets, (
            f"Expected {self.n_assets} assets, got {P}"
        )

        # Flatten asset × feature dims → [B, Seq, P*F]
        x_flat = x.reshape(B, S, P * F)

        # LSTM encoding — use only the final hidden state
        lstm_out, _ = self.lstm(x_flat)        # [B, Seq, H]
        h_last = lstm_out[:, -1, :]            # [B, H]
        h_last = self.layer_norm(h_last)       # [B, H]

        # Project to Cholesky vector
        chol_vec = self.fc(h_last)             # [B, cholesky_size]

        # ── Reconstruct lower-triangular L ────────
        L = torch.zeros(B, self.n_assets, self.n_assets,
                        device=x.device, dtype=x.dtype)
        L[:, self.tril_row, self.tril_col] = chol_vec

        # Exponentiate diagonal to guarantee strict positivity
        # Financial intuition: the diagonal of L controls marginal volatilities;
        # exp() ensures they can never go negative.
        diag_idx = torch.arange(self.n_assets, device=x.device)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx])

        # ── Σ̂ = L · Lᵀ + ε · I ──────────────────
        # The epsilon term prevents numerical singularity during
        # backpropagation through the cvxpylayers solver.
        sigma = torch.bmm(L, L.transpose(1, 2))
        sigma = sigma + config.EPSILON * torch.eye(
            self.n_assets, device=x.device, dtype=x.dtype
        ).unsqueeze(0)

        return sigma

    def get_cholesky_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the raw Cholesky factor L (before Σ = LLᵀ).

        Useful for diagnostic logging and SHAP attribution.
        """
        B, S, P, F = x.shape
        x_flat = x.reshape(B, S, P * F)
        lstm_out, _ = self.lstm(x_flat)
        h_last = self.layer_norm(lstm_out[:, -1, :])
        chol_vec = self.fc(h_last)

        L = torch.zeros(B, self.n_assets, self.n_assets,
                        device=x.device, dtype=x.dtype)
        L[:, self.tril_row, self.tril_col] = chol_vec
        diag_idx = torch.arange(self.n_assets, device=x.device)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx])
        return L
