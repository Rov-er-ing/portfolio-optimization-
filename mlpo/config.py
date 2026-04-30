"""
MLPO v1.0 — Centralized Configuration Singleton.

Every hyperparameter, dimensionality, and constraint lives here.
NO MAGIC NUMBERS anywhere else in the codebase.

Hardware Target: Lenovo LOQ — NVIDIA RTX 3050 (6GB VRAM), 16GB RAM.
"""

import torch
import numpy as np
import random
import os
from typing import List


class Config:
    """
    Singleton configuration for the ML-Powered Portfolio Optimization System.

    Attributes
    ----------
    DEVICE : torch.device
        Auto-detected compute device (CUDA / CPU).
    ASSET_LIST : List[str]
        Full 45-instrument universe.
    SEQUENCE_LENGTH : int
        LSTM lookback window, capped at 60 for VRAM budget.
    """

    # ──────────────────────────────────────────────
    # Asset Universe  (45 instruments)
    # ──────────────────────────────────────────────
    EQUITIES: List[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "NVDA", "META", "BRK-B", "UNH", "JNJ",
        "XOM", "V", "JPM", "TSM", "PG",
        "MA", "AVGO", "HD", "CVX", "LLY",
        "ABBV", "MRK", "PFE", "COST", "PEP",
        "KO", "TMO", "WMT", "MCD", "DIS",
    ]
    FIXED_INCOME: List[str] = [
        "TLT", "IEF", "SHY", "LQD", "HYG",
        "BND", "TIP", "AGG", "EMB", "MUB",
    ]
    COMMODITIES: List[str] = ["GLD", "SLV", "USO", "UNG", "DBC"]

    ASSET_LIST: List[str] = EQUITIES + FIXED_INCOME + COMMODITIES
    N_ASSETS: int = len(ASSET_LIST)  # 45

    # Defensive asset indices (Fixed Income + Commodities) — used by VIX override
    DEFENSIVE_INDICES: List[int] = list(range(len(EQUITIES), N_ASSETS))

    # ──────────────────────────────────────────────
    # Macro State Variables  (6 variables)
    # ──────────────────────────────────────────────
    MACRO_STATE_VARIABLES: List[str] = [
        "^VIX", "TEDrate", "T10Y2Y",
        "SPY_MOM", "M2_GROWTH", "MARKET_CAP",
    ]
    N_MACRO: int = len(MACRO_STATE_VARIABLES)  # 6

    # ──────────────────────────────────────────────
    # LSTM Volatility Forecaster  (Module A)
    # ──────────────────────────────────────────────
    LSTM_HIDDEN: int = 256      # Hidden units per LSTM layer
    LSTM_LAYERS: int = 3        # Stacked LSTM depth
    LSTM_DROPOUT: float = 0.3   # Inter-layer dropout
    N_FEATURES: int = 2         # z_score + volatility per asset
    FEATURE_NAMES: List[str] = ["z_score", "volatility"]

    # ──────────────────────────────────────────────
    # Regime Detection  (Module B)
    # ──────────────────────────────────────────────
    REGIME_LSTM_HIDDEN: int = 128   # Encoder hidden units
    REGIME_LSTM_LAYERS: int = 2     # Encoder depth
    N_REGIMES: int = 3              # GMM components (bull, neutral, bear)

    # ──────────────────────────────────────────────
    # Training Hyperparameters
    # ──────────────────────────────────────────────
    SEQUENCE_LENGTH: int = 60   # Capped for 6GB VRAM budget
    BATCH_SIZE: int = 16        # Restricted for cvxpylayers memory
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5  # L2 regularisation coefficient
    EPOCHS: int = 100
    GRAD_CLIP_NORM: float = 1.0  # Gradient clipping max norm

    # ──────────────────────────────────────────────
    # Mathematical / Financial Constants
    # ──────────────────────────────────────────────
    EPSILON: float = 1e-6       # Numerical stability for Cholesky + backprop
    RISK_FREE_RATE: float = 0.02  # Annualised risk-free rate
    TRADING_DAYS: int = 252     # Annualisation factor

    # ──────────────────────────────────────────────
    # Portfolio Constraints
    # ──────────────────────────────────────────────
    MAX_WEIGHT: float = 0.15    # Upper bound on single-asset weight
    VIX_THRESHOLD: float = 35.0  # Tail-risk override trigger
    DEFENSIVE_ALLOC: float = 0.40  # Min defensive allocation when VIX >= threshold

    # Sparse Attention parameters
    ATTENTION_K: int = 15       # k-nearest correlated neighbors

    # ──────────────────────────────────────────────
    # Composite Loss Weights
    # ──────────────────────────────────────────────
    LOSS_GAMMA_MDD: float = 0.5    # Max drawdown penalty
    LOSS_GAMMA_TURNOVER: float = 0.1  # Turnover penalty
    LOSS_GAMMA_L2: float = 1e-4    # Parameter L2 regularisation

    # ──────────────────────────────────────────────
    # Optuna Hyperparameter Search
    # ──────────────────────────────────────────────
    OPTUNA_N_TRIALS: int = 50   # Reduced from 100 for single-GPU budget
    OPTUNA_PRUNE_EPOCH: int = 20  # Prune below-median trials after this epoch

    # ──────────────────────────────────────────────
    # Hardware Configuration
    # ──────────────────────────────────────────────
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    USE_AMP: bool = torch.cuda.is_available()

    # ──────────────────────────────────────────────
    # Walk-Forward Backtesting
    # ──────────────────────────────────────────────
    EMBARGO_DAYS: int = 21      # Mandatory gap between train and val/test
    TRANSACTION_COST_BPS: float = 10.0  # 10 bps per trade

    # ──────────────────────────────────────────────
    # Paths
    # ──────────────────────────────────────────────
    DATA_DIR: str = "data_cache"
    MODEL_DIR: str = "model_artifacts"

    def __init__(self) -> None:
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    """
    Set deterministic seeds across all RNG sources for reproducible experiments.

    Parameters
    ----------
    seed : int
        The seed value to use everywhere.

    Notes
    -----
    `torch.backends.cudnn.benchmark = False` sacrifices ~5% speed for
    bitwise reproducibility — acceptable on consumer hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Singleton Instance ────────────────────────────
config = Config()
