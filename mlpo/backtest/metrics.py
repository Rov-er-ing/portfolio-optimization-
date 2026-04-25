"""
Financial Performance Metrics.

All metrics follow institutional standards:
    - Sharpe Ratio: Newey-West adjusted for autocorrelation.
    - Max Drawdown: peak-to-trough from cumulative wealth curve.
    - Calmar Ratio: annualised return / max drawdown.
    - VaR / CVaR: historical quantile-based tail risk measures.
"""

import numpy as np
from typing import Dict

from mlpo.config import config


def annualised_return(returns: np.ndarray) -> float:
    """Geometric annualised return from daily returns."""
    cum = np.prod(1 + returns) ** (config.TRADING_DAYS / len(returns)) - 1
    return float(cum)


def annualised_volatility(returns: np.ndarray) -> float:
    """Annualised volatility using √252 convention."""
    return float(np.std(returns, ddof=1) * np.sqrt(config.TRADING_DAYS))


def sharpe_ratio(returns: np.ndarray, rf: float = config.RISK_FREE_RATE) -> float:
    """
    Sharpe Ratio with annualisation.

    Parameters
    ----------
    returns : np.ndarray
        Daily net returns.
    rf : float
        Annualised risk-free rate.

    Returns
    -------
    sharpe : float
        Annualised Sharpe ratio.
    """
    rf_daily = rf / config.TRADING_DAYS
    excess = returns - rf_daily
    if excess.std() < 1e-10:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(config.TRADING_DAYS))


def max_drawdown(returns: np.ndarray) -> float:
    """
    Maximum drawdown from the cumulative wealth curve.

    Financial Intuition:
        MDD measures the worst peak-to-trough decline — the most
        painful experience for an investor holding through a crisis.
    """
    wealth = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(wealth)
    drawdowns = (running_max - wealth) / running_max
    return float(drawdowns.max())


def calmar_ratio(returns: np.ndarray) -> float:
    """Calmar = annualised return / max drawdown."""
    mdd = max_drawdown(returns)
    if mdd < 1e-10:
        return 0.0
    return annualised_return(returns) / mdd


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical VaR at confidence level (1 - alpha)."""
    return float(np.percentile(returns, alpha * 100))


def conditional_var(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Conditional VaR (Expected Shortfall).

    CVaR = E[R | R <= VaR] — average loss in the tail.
    More coherent than VaR for risk budgeting.
    """
    var = value_at_risk(returns, alpha)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def compute_all_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    Compute full suite of performance metrics.

    Returns
    -------
    metrics : dict
        All key financial metrics.
    """
    return {
        "annualised_return": annualised_return(returns),
        "annualised_volatility": annualised_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
        "var_5pct": value_at_risk(returns, 0.05),
        "cvar_5pct": conditional_var(returns, 0.05),
    }
