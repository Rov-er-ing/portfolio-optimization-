from typing import Generator, Tuple
import numpy as np
from mlpo.config import config

class WalkForwardSplitter:
    """
    Chronological Walk-Forward Splitter with Embargo.
    
    Generates training and validation folds that move sequentially forward
    in time. Injects a strict embargo gap between training and validation
    to prevent lookahead bias.
    """
    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 252, # 1 year of trading days
        val_size: int = 63,    # 3 months of trading days
        embargo_days: int = config.EMBARGO_DAYS
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.embargo_days = embargo_days
        
    def split(self, total_length: int) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train and val indices.
        
        Parameters
        ----------
        total_length : int
            Total number of temporal steps in the dataset.
            
        Yields
        ------
        train_idx : np.ndarray
        val_idx : np.ndarray
        """
        required_length = self.train_size + self.embargo_days + self.val_size
        if total_length < required_length:
            raise ValueError(f"Dataset length {total_length} is too short. Requires at least {required_length}.")
            
        step_size = (total_length - required_length) // max(1, self.n_splits - 1)
        
        for i in range(self.n_splits):
            start_idx = i * step_size
            train_end = start_idx + self.train_size
            val_start = train_end + self.embargo_days
            val_end = val_start + self.val_size
            
            # Ensure we don't go out of bounds (can happen due to integer division on step_size)
            if val_end > total_length:
                break
                
            train_idx = np.arange(start_idx, train_end)
            val_idx = np.arange(val_start, val_end)
            
            yield train_idx, val_idx
