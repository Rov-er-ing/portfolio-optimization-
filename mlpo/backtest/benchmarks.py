"""
Benchmark Strategies.

Provides baseline portfolios for comparison:
    - Equal Weight (1/N)
    - Risk Parity (inverse volatility)

These serve as the "bar to clear" for the MLPO model.
"""

import numpy as np
from typing import Dict

from mlpo.config import config


def equal_weight_portfolio(n_assets: int = config.N_ASSETS) -> np.ndarray:
    """1/N portfolio — surprisingly hard to beat in practice."""
    return np.ones(n_assets) / n_assets


def inverse_volatility_weights(
    returns: np.ndarray,
    lookback: int = 60,
) -> np.ndarray:
    """
    Risk Parity via inverse-volatility weighting.

    Financial Intuition:
        Allocate less to volatile assets, more to stable ones.
        This is the simplest risk-parity heuristic; full risk parity
        would also account for correlations.

    Parameters
    ----------
    returns : np.ndarray
        Shape ``[T, P]`` — historical returns for lookback estimation.
    lookback : int
        Number of trailing days for volatility estimation.

    Returns
    -------
    weights : np.ndarray
        Shape ``[P]`` — normalised inverse-volatility weights.
    """
    recent = returns[-lookback:]
    vols = np.std(recent, axis=0, ddof=1)

    # Avoid division by zero for constant-price assets
    vols = np.maximum(vols, 1e-10)

    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return weights


def generate_benchmark_weights(
    returns_history: np.ndarray,
    method: str = "equal_weight",
    lookback: int = 60,
) -> np.ndarray:
    """
    Generate time-series of benchmark weights.

    Parameters
    ----------
    returns_history : np.ndarray
        Shape ``[T, P]``.
    method : str
        'equal_weight' or 'risk_parity'.
    lookback : int
        Lookback for risk parity.

    Returns
    -------
    weights_history : np.ndarray
        Shape ``[T, P]``.
    """
    T, P = returns_history.shape
    weights = np.zeros((T, P))

    for t in range(T):
        if method == "equal_weight":
            weights[t] = equal_weight_portfolio(P)
        elif method == "risk_parity":
            if t < lookback:
                weights[t] = equal_weight_portfolio(P)
            else:
                weights[t] = inverse_volatility_weights(
                    returns_history[:t], lookback,
                )
        else:
            raise ValueError(f"Unknown benchmark method: {method}")

    return weights
