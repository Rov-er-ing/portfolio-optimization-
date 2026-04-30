import torch
from mlpo.training.splitter import WalkForwardSplitter
from mlpo.config import config
from mlpo.training.trainer import Trainer
from mlpo.models.portfolio import PortfolioOptimizer

def test_walk_forward_splitter_embargo():
    """
    Test the WalkForwardSplitter to ensure the 21-day embargo constraint is strictly honored.
    """
    total_length = 1000
    splitter = WalkForwardSplitter(n_splits=3, train_size=200, val_size=50, embargo_days=config.EMBARGO_DAYS)
    
    splits = list(splitter.split(total_length))
    
    assert len(splits) == 3, "Expected 3 splits"
    
    for train_idx, val_idx in splits:
        train_end = train_idx[-1]
        val_start = val_idx[0]
        # Verify the exact embargo period
        assert val_start - train_end == config.EMBARGO_DAYS + 1, f"Embargo violated: {val_start} - {train_end} != {config.EMBARGO_DAYS} + 1"
        
def test_trainer_amp_initialization():
    """
    Test Trainer initialization and AMP scaler presence to ensure
    memory optimization setup is valid.
    """
    model = PortfolioOptimizer(
        n_assets=config.N_ASSETS,
        n_macro=config.N_MACRO,
        n_features=config.N_FEATURES
    )
    
    trainer = Trainer(model=model, use_amp=True)
    
    assert trainer.use_amp is True, "Trainer use_amp should be True"
    assert hasattr(trainer, 'scaler'), "Trainer missing GradScaler"
    if torch.cuda.is_available():
        assert trainer.scaler.is_enabled() is True, "AMP Scaler should be enabled if CUDA is available"
