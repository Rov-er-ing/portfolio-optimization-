import os
import logging
import torch

from mlpo.data.ingestion import get_data_pipeline
from mlpo.data.features import prepare_feature_tensor
from mlpo.data.dataset import create_dataloaders
from mlpo.models.portfolio import PortfolioOptimizer
from mlpo.training.trainer import Trainer
from mlpo.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Clear old bad cache if it exists
    cache_path = os.path.join(config.DATA_DIR, "raw_prices.csv")
    if os.path.exists(cache_path):
        logger.info(f"Deleting old cache: {cache_path}")
        os.remove(cache_path)

    # 2. Ingest Data (Downloads from Yahoo Finance)
    logger.info("Starting Data Ingestion...")
    prices = get_data_pipeline()
    
    if prices.empty:
        logger.error("Failed to fetch prices. Exiting.")
        return

    # 3. Engineer Features
    logger.info("Computing Rolling Features...")
    features, valid_mask = prepare_feature_tensor(prices)
    
    # 4. Target returns (Next day returns)
    returns = prices.pct_change().shift(-1).dropna()
    
    # Ensure targets and features align
    common_idx = features['z_score'].index.intersection(returns.index)
    features = {k: v.loc[common_idx] for k, v in features.items()}
    targets = returns.loc[common_idx]
    
    # 5. Create DataLoaders
    logger.info("Setting up DataLoaders...")
    train_loader, val_loader = create_dataloaders(features, targets, batch_size=config.BATCH_SIZE)
    
    # 6. Initialize Model & Trainer
    logger.info(f"Initializing Portfolio Optimizer on {config.DEVICE}...")
    model = PortfolioOptimizer()
    trainer = Trainer(model=model, use_amp=config.USE_AMP)
    
    # 7. Train Model
    logger.info("Starting Walk-Forward Training...")
    history = trainer.fit(train_loader, val_loader, epochs=5)
    
    logger.info("Training complete! The best model weights are saved in the `model_artifacts` directory.")
    logger.info("You can now start the API server by running: python -m uvicorn mlpo.api.main:app --reload")

if __name__ == "__main__":
    main()
