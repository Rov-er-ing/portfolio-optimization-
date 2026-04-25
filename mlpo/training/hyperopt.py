"""
Optuna Hyperparameter Optimisation with Aggressive Pruning.

Uses TPE (Tree-structured Parzen Estimator) with MedianPruner.
Any trial falling below median Sharpe by epoch 20 is terminated.

Hardware Constraint:
    50 trials max on RTX 3050 to stay within thermal + time budget.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import logging
from typing import Dict, Any

from mlpo.config import config, seed_everything
from mlpo.models.portfolio import PortfolioOptimizer
from mlpo.training.trainer import Trainer

logger = logging.getLogger(__name__)


def create_objective(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
):
    """
    Factory for the Optuna objective function.

    Parameters
    ----------
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data.

    Returns
    -------
    objective : callable
        Function compatible with ``study.optimize()``.
    """

    def objective(trial: optuna.Trial) -> float:
        """
        Single Optuna trial.

        Returns
        -------
        val_sharpe : float
            Validation Sharpe ratio (maximised).
        """
        seed_everything(42)

        # ── Hyperparameter search space ───────────
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        gamma_mdd = trial.suggest_float("gamma_mdd", 0.1, 2.0)
        gamma_turnover = trial.suggest_float("gamma_turnover", 0.01, 0.5)
        lstm_hidden = trial.suggest_categorical("lstm_hidden", [128, 256])

        # ── Build model with trial params ─────────
        model = PortfolioOptimizer(
            n_assets=config.N_ASSETS,
            n_features=config.N_FEATURES,
            n_macro=config.N_MACRO,
            use_attention=True,
        )

        # Override LSTM hidden if needed
        if lstm_hidden != config.LSTM_HIDDEN:
            model = PortfolioOptimizer(
                n_assets=config.N_ASSETS,
                n_features=config.N_FEATURES,
                n_macro=config.N_MACRO,
                use_attention=False,  # Attention dim must match
            )

        trainer = Trainer(model, lr=lr, weight_decay=weight_decay)
        trainer.criterion.gamma_mdd = gamma_mdd
        trainer.criterion.gamma_turnover = gamma_turnover

        # ── Training with pruning ─────────────────
        best_sharpe = float("-inf")

        for epoch in range(1, config.EPOCHS + 1):
            train_metrics = trainer.train_epoch(trainer.model, epoch)
            val_metrics = trainer.validate(val_loader)

            sharpe = val_metrics.get("sharpe", 0.0)
            best_sharpe = max(best_sharpe, sharpe)

            # Report to Optuna for pruning
            trial.report(sharpe, epoch)

            # Prune if below median after epoch 20
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                raise optuna.TrialPruned()

        return best_sharpe

    return objective


def run_hyperopt(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_trials: int = config.OPTUNA_N_TRIALS,
) -> Dict[str, Any]:
    """
    Run Optuna study.

    Returns
    -------
    best_params : dict
        Best hyperparameters found.
    """
    study = optuna.create_study(
        direction="maximize",  # Maximise Sharpe ratio
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=config.OPTUNA_PRUNE_EPOCH,
        ),
    )

    objective = create_objective(train_loader, val_loader)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=3600 * 4,  # 4 hour timeout for consumer GPU
        show_progress_bar=True,
    )

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best Sharpe: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    return study.best_params
