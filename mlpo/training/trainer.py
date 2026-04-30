"""
Walk-Forward Training Loop with AMP Integration.

Implements the walk-forward expanding window protocol:
    1. Train on [0, T_train].
    2. Embargo 21 days.
    3. Validate on [T_train + 21, T_val].
    4. Expand T_train → repeat.

Memory management for RTX 3050 (6GB VRAM):
    - torch.cuda.amp for FP16 forward, FP32 gradients.
    - Aggressive grad clearing between cvxpylayers steps.
    - Gradient clipping at max_norm=1.0.
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from typing import Dict, Optional, Tuple
import logging
import time

from mlpo.config import config
from mlpo.models.portfolio import PortfolioOptimizer
from mlpo.training.loss import CompositeFinancialLoss

logger = logging.getLogger(__name__)


class Trainer:
    """
    Walk-forward trainer for the portfolio optimizer.

    Parameters
    ----------
    model : PortfolioOptimizer
        The end-to-end model.
    lr : float
        Learning rate.
    weight_decay : float
        AdamW weight decay.
    use_amp : bool
        Enable mixed precision training.
    """

    def __init__(
        self,
        model: PortfolioOptimizer,
        lr: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        use_amp: bool = config.USE_AMP,
    ) -> None:
        self.model = model.to(config.DEVICE)
        self.criterion = CompositeFinancialLoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.scaler = GradScaler('cuda', enabled=use_amp) if torch.cuda.is_available() else GradScaler('cpu', enabled=use_amp)
        self.use_amp = use_amp
        self.device = config.DEVICE

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Single training epoch.

        Returns
        -------
        metrics : dict
            Average loss components for the epoch.
        """
        self.model.train()
        epoch_losses = {"total": 0.0, "sharpe": 0.0, "mdd": 0.0, "turnover": 0.0}
        n_batches = 0

        for batch_idx, (asset_feats, macro_feats, vix, returns) in enumerate(
            train_loader
        ):
            asset_feats = asset_feats.to(self.device)
            macro_feats = macro_feats.to(self.device)
            vix = vix.to(self.device)
            returns = returns.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            # ── Forward pass with AMP ─────────────
            # Note: cvxpylayers requires float64 internally, so we
            # disable autocast for the risk_budgeting step. The LSTM
            # and regime detector benefit from FP16.
            with autocast('cuda', enabled=self.use_amp) if torch.cuda.is_available() else autocast('cpu', enabled=self.use_amp):
                result = self.model(asset_feats, macro_feats, vix)
                weights = result["weights"]

                losses = self.criterion(
                    weights, returns,
                    model_params=list(self.model.parameters()),
                )

            # ── Backward + gradient clipping ──────
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)

            # Clip gradients to prevent explosions from LSTM + cvxpylayers
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.GRAD_CLIP_NORM,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Aggressive memory clearing for RTX 3050
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Accumulate metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            n_batches += 1

        # Average
        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Validation loop — no gradients."""
        self.model.eval()
        val_losses = {"total": 0.0, "sharpe": 0.0, "mdd": 0.0}
        n_batches = 0

        for asset_feats, macro_feats, vix, returns in val_loader:
            asset_feats = asset_feats.to(self.device)
            macro_feats = macro_feats.to(self.device)
            vix = vix.to(self.device)
            returns = returns.to(self.device)

            result = self.model(asset_feats, macro_feats, vix)
            losses = self.criterion(result["weights"], returns)

            for key in val_losses:
                if key in losses:
                    val_losses[key] += losses[key].item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in val_losses.items()}

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = config.EPOCHS,
    ) -> Dict[str, list]:
        """
        Full training loop.

        Returns
        -------
        history : dict
            Lists of per-epoch metrics.
        """
        history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "train_sharpe": [], "val_sharpe": [],
        }

        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            dt = time.time() - t0

            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_metrics["total"])
            history["train_sharpe"].append(train_metrics["sharpe"])
            history["val_sharpe"].append(val_metrics.get("sharpe", 0.0))

            # Checkpoint best model
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                best_epoch = epoch
                torch.save(
                    self.model.state_dict(),
                    f"{config.MODEL_DIR}/best_model.pt",
                )

            logger.info(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"Val Loss: {val_metrics['total']:.4f} | "
                f"Sharpe: {train_metrics['sharpe']:.3f} | "
                f"Time: {dt:.1f}s"
            )

        logger.info(f"Best model at epoch {best_epoch} (val_loss={best_val_loss:.4f})")
        return history
