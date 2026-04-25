import torch
import os

class Config:
    # Asset Universe (45 Instruments)
    EQUITIES = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "UNH", "JNJ",
        "XOM", "V", "JPM", "TSM", "PG", "MA", "AVGO", "HD", "CVX", "LLY",
        "ABBV", "MRK", "PFE", "COST", "PEP", "KO", "TMO", "WMT", "MCD", "DIS"
    ]
    FIXED_INCOME = ["TLT", "IEF", "SHY", "LQD", "HYG", "BND", "TIP", "AGG", "EMB", "MUB"]
    COMMODITIES = ["GLD", "SLV", "USO", "UNG", "DBC"]
    
    ASSET_LIST = EQUITIES + FIXED_INCOME + COMMODITIES
    
    # Macro State Variables (6 Variables)
    MACRO_STATE_VARIABLES = ["^VIX", "TEDrate", "T10Y2Y", "SPY_MOM", "M2_GROWTH", "MARKET_CAP"]
    
    # Training Hyperparameters
    SEQUENCE_LENGTH = 60  # Capped for VRAM (4GB-6GB)
    BATCH_SIZE = 16       # Restricted for cvxpylayers VRAM usage
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    
    # Mathematical Constraints
    EPSILON = 1e-6        # For numerical stability in Cholesky/Backprop
    RISK_FREE_RATE = 0.02 # Assumed annual RF rate
    
    # Optimization Constraints
    MAX_WEIGHT = 0.15     # 0 <= w_i <= 0.15
    VIX_THRESHOLD = 35    # Tail-risk override trigger
    DEFENSIVE_ALLOC = 0.40 # Min 40% defensive allocation if VIX >= threshold
    
    # Hardware Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = torch.cuda.is_available() # Use Automatic Mixed Precision if GPU is available
    
    # Data Paths
    DATA_DIR = "data_cache"
    os.makedirs(DATA_DIR, exist_ok=True)

# Singleton Instance
config = Config()
