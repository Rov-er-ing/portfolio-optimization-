"""
Walk-Forward Backtest Engine.

Simulates portfolio performance using walk-forward expanding windows
with transaction cost modelling.

Financial Standards:
    - Transaction costs: 10 bps per trade (configurable).
    - All metrics computed on NET returns (after costs).
    - 21-day embargo between train/val boundaries.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from mlpo.config import config

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Walk-forward portfolio backtester.

    Parameters
    ----------
    transaction_cost_bps : float
        One-way transaction cost in basis points.
    """

    def __init__(
        self,
        transaction_cost_bps: float = config.TRANSACTION_COST_BPS,
    ) -> None:
        self.tc = transaction_cost_bps / 10_000  # Convert bps → decimal

    def run(
        self,
        weights_history: np.ndarray,
        returns_history: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Execute backtest.

        Parameters
        ----------
        weights_history : np.ndarray
            Shape ``[T, P]`` — portfolio weights at each rebalance.
        returns_history : np.ndarray
            Shape ``[T, P]`` — asset returns at each period.
        dates : DatetimeIndex, optional
            Date labels for the result series.

        Returns
        -------
        results : dict
            'portfolio_returns': [T] net returns.
            'cumulative_wealth': [T] cumulative NAV.
            'weights': [T, P] realised weights.
            'turnover': [T] per-period turnover.
            'costs': [T] transaction costs incurred.
        """
        T, P = weights_history.shape
        assert returns_history.shape == (T, P), (
            f"Shape mismatch: weights={weights_history.shape}, "
            f"returns={returns_history.shape}"
        )

        port_returns = np.zeros(T)
        turnover = np.zeros(T)
        costs = np.zeros(T)

        w_prev = np.ones(P) / P  # Start equal-weight

        for t in range(T):
            w_target = weights_history[t]

            # ── Turnover = Σ |w_target - w_prev| ──
            turn = np.abs(w_target - w_prev).sum()
            turnover[t] = turn

            # ── Transaction cost (proportional) ───
            # Financial intuition: each unit of turnover incurs
            # the cost of both selling old and buying new positions.
            cost = turn * self.tc
            costs[t] = cost

            # ── Portfolio return (gross - costs) ──
            gross_return = np.dot(w_target, returns_history[t])
            port_returns[t] = gross_return - cost

            # ── Drift weights by returns for next period ──
            # After returns, weights drift from target due to
            # differential asset performance.
            w_drifted = w_target * (1 + returns_history[t])
            w_prev = w_drifted / (w_drifted.sum() + 1e-10)

        cumulative = np.cumprod(1 + port_returns)

        return {
            "portfolio_returns": port_returns,
            "cumulative_wealth": cumulative,
            "weights": weights_history,
            "turnover": turnover,
            "costs": costs,
        }
