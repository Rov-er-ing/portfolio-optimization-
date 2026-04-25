"""
Module D — Sparse Attention Mechanism.

Standard self-attention is O(p²) over p assets. By restricting each
asset's attention to its k most correlated neighbors, complexity drops
to O(p·k) where k << p.

The neighborhood graph is pre-computed from historical correlation
matrices and registered as a buffer (no gradient, moves with device).

Financial Intuition:
    Assets within the same sector or macro-factor loading tend to
    co-move.  Sparse attention captures these cluster dynamics while
    avoiding the noise from weakly-correlated pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from mlpo.config import config


class SparseAttention(nn.Module):
    """
    O(p·k) Sparse Multi-Head Attention over asset embeddings.

    Parameters
    ----------
    embed_dim : int
        Dimension of each asset embedding.
    n_heads : int
        Number of attention heads.
    k : int
        Number of correlated neighbors per asset.
    n_assets : int
        Total assets in the universe.
    """

    def __init__(
        self,
        embed_dim: int = config.LSTM_HIDDEN,
        n_heads: int = 4,
        k: int = config.ATTENTION_K,
        n_assets: int = config.N_ASSETS,
    ) -> None:
        super().__init__()
        assert embed_dim % n_heads == 0, (
            f"embed_dim={embed_dim} must be divisible by n_heads={n_heads}"
        )

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.k = k
        self.n_assets = n_assets

        # Q, K, V projections
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)

        # Neighborhood mask: [N_assets, N_assets] boolean
        # True at (i, j) means asset j is in asset i's neighborhood
        self.register_buffer(
            "neighbor_mask",
            torch.zeros(n_assets, n_assets, dtype=torch.bool),
        )

    def set_neighborhoods(
        self,
        neighborhoods: Dict[str, List[str]],
        asset_list: List[str],
    ) -> None:
        """
        Build the sparse attention mask from pre-computed correlations.

        Parameters
        ----------
        neighborhoods : dict
            {asset_name: [neighbor1, neighbor2, ...]} from features.py.
        asset_list : list
            Ordered asset names matching the tensor's asset dimension.
        """
        idx_map = {name: i for i, name in enumerate(asset_list)}
        mask = torch.zeros(self.n_assets, self.n_assets, dtype=torch.bool)

        for asset, neighbors in neighborhoods.items():
            if asset not in idx_map:
                continue
            i = idx_map[asset]
            mask[i, i] = True  # Self-attention always allowed
            for nb in neighbors[: self.k]:
                if nb in idx_map:
                    mask[i, idx_map[nb]] = True

        self.neighbor_mask.copy_(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse attention forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Asset embeddings, shape ``[B, P, D]`` where P=n_assets.

        Returns
        -------
        out : torch.Tensor
            Attended embeddings, shape ``[B, P, D]``.
        """
        B, P, D = x.shape
        assert P == self.n_assets, f"Expected {self.n_assets} assets, got {P}"
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        # ── Q, K, V projections ───────────────────
        Q = self.W_q(x).reshape(B, P, self.n_heads, self.head_dim)
        K = self.W_k(x).reshape(B, P, self.n_heads, self.head_dim)
        V = self.W_v(x).reshape(B, P, self.n_heads, self.head_dim)

        # Transpose to [B, Heads, P, D_head]
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # ── Scaled dot-product attention ──────────
        scale = self.head_dim ** 0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, P, P]

        # ── Apply sparse mask ─────────────────────
        # Where mask is False, set score to -inf → softmax → 0
        # Financial intuition: uncorrelated asset pairs contribute no
        # attention weight, preserving the sparse interaction structure.
        mask = self.neighbor_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, P, P]
        scores = scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, P, P]
        # Replace NaN from all-masked rows with zeros
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # ── Weighted sum ──────────────────────────
        context = torch.matmul(attn_weights, V)  # [B, H, P, D_head]
        context = context.permute(0, 2, 1, 3).reshape(B, P, D)

        return self.W_o(context)
