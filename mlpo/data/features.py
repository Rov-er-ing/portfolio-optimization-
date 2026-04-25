import pandas as pd
import numpy as np
from mlpo.config import config
import logging

logger = logging.getLogger(__name__)

def compute_rolling_features(prices, window=21):
    """
    Computes rolling returns, volatility, and Z-scores.
    Enforces .shift(1) to prevent lookahead bias.
    """
    # 1. Daily Returns
    returns = prices.pct_change()
    
    # 2. Shift returns to ensure features at time t use data only up to t-1
    lagged_returns = returns.shift(1)
    
    # 3. Rolling Volatility (annualized)
    volatility = lagged_returns.rolling(window=window).std() * np.sqrt(252)
    
    # 4. Rolling Z-scores of returns
    mean_ret = lagged_returns.rolling(window=window).mean()
    std_ret = lagged_returns.rolling(window=window).std()
    z_scores = (lagged_returns - mean_ret) / (std_ret + 1e-8)
    
    # Concatenate features (e.g., Z-score and Volatility)
    # This is a simplified feature set; in a real scenario, we'd include macro variables too.
    features = {
        'z_score': z_scores,
        'volatility': volatility
    }
    
    return features

def compute_neighborhood_correlations(prices, k=15):
    """
    Computes the k-most correlated neighbors for each asset.
    Used for the Sparse Attention mechanism in Phase 3.
    """
    returns = prices.pct_change().dropna()
    corr_matrix = returns.corr()
    
    neighborhoods = {}
    for asset in corr_matrix.columns:
        # Get top k correlated assets (excluding itself)
        top_k = corr_matrix[asset].sort_values(ascending=False)[1:k+1].index.tolist()
        neighborhoods[asset] = top_k
        
    return neighborhoods

def prepare_feature_tensor(prices):
    """
    Unified pipeline to generate the feature set.
    """
    features = compute_rolling_features(prices)
    # Drop rows with NaNs from rolling windows
    valid_mask = features['z_score'].notna().all(axis=1) & features['volatility'].notna().all(axis=1)
    
    processed_features = {k: v[valid_mask] for k, v in features.items()}
    return processed_features, valid_mask

if __name__ == "__main__":
    # Test with dummy data
    dates = pd.date_range("2020-01-01", periods=100)
    dummy_prices = pd.DataFrame(np.random.randn(100, 5), index=dates, columns=['A', 'B', 'C', 'D', 'E'])
    feats, mask = prepare_feature_tensor(dummy_prices)
    print(f"Features generated. Shape: {feats['z_score'].shape}")
