import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from mlpo.config import config

class MLPODataset(Dataset):
    def __init__(self, feature_dfs, target_df, seq_length=60, embargo=21):
        """
        PyTorch Dataset for Portfolio Optimization.
        
        Args:
            feature_dfs (dict): Dictionary of DataFrames (e.g., {'z_score': df, 'vol': df})
            target_df (DataFrame): The future returns (targets) - NOT lagged.
            seq_length (int): LSTM sequence length.
            embargo (int): Gap between training and test sets.
        """
        self.seq_length = seq_length
        self.embargo = embargo
        
        # Convert DataFrames to Tensors
        # Shape: [N_Assets, N_Timesteps, N_Features]
        self.z_score = torch.tensor(feature_dfs['z_score'].values, dtype=torch.float32)
        self.vol = torch.tensor(feature_dfs['volatility'].values, dtype=torch.float32)
        self.targets = torch.tensor(target_df.values, dtype=torch.float32)
        
        self.n_timesteps = self.z_score.shape[0]
        self.n_assets = self.z_score.shape[1]
        
    def __len__(self):
        return self.n_timesteps - self.seq_length
    
    def __getitem__(self, idx):
        """
        Returns:
            X: [Seq_Len, N_Assets, N_Features]
            y: [N_Assets] (The return at t+1)
        """
        # Feature window [t-seq_len : t]
        x_z = self.z_score[idx : idx + self.seq_length] # [Seq, Assets]
        x_v = self.vol[idx : idx + self.seq_length]     # [Seq, Assets]
        
        # Stack features: [Seq, Assets, Feats]
        X = torch.stack([x_z, x_v], dim=-1)
        
        # Target at time t+1 (idx + seq_length)
        # Note: In a real walk-forward, we'd be careful about the indexing relative to features
        y = self.targets[idx + self.seq_length]
        
        # Create dummy macro and VIX features for now
        # Shape: macro_feats [Seq_Len, N_Macro], VIX [1]
        macro_feats = torch.zeros(self.seq_length, config.N_MACRO, dtype=torch.float32)
        vix = torch.tensor([20.0], dtype=torch.float32)  # Default VIX
        
        return X, macro_feats, vix, y

def create_dataloaders(feature_dfs, target_df, batch_size=None):
    if batch_size is None:
        batch_size = config.BATCH_SIZE
        
    dataset = MLPODataset(feature_dfs, target_df, seq_length=config.SEQUENCE_LENGTH)
    
    # Simple split for demonstration (last 20% is val)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test stub
    pass
