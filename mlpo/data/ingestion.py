import yfinance as yf
import pandas as pd
import os
from mlpo.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_historical_data(tickers, start_date="2010-01-01", end_date=None):
    """
    Fetches adjusted closing prices for a list of tickers.
    """
    logger.info(f"Fetching data for {len(tickers)} tickers from {start_date}...")
    
    # yfinance often works better with '.' instead of '-' for some tickers
    sanitized_tickers = [t.replace("-", ".") for t in tickers]
    
    try:
        data = yf.download(sanitized_tickers, start=start_date, end=end_date, auto_adjust=True)
        # We primarily need adjusted Close
        if 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
            
        # Map back to original tickers if needed
        prices.columns = tickers
        
        return prices
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def get_data_pipeline():
    """
    Main entry point for data ingestion. 
    Checks cache first, then downloads.
    """
    cache_path = os.path.join(config.DATA_DIR, "raw_prices.csv")
    
    if os.path.exists(cache_path):
        logger.info("Loading data from cache...")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)
    
    # Fetch assets
    prices = fetch_historical_data(config.ASSET_LIST)
    
    # Save to cache
    if not prices.empty:
        prices.to_csv(cache_path)
        logger.info(f"Data cached to {cache_path}")
        
    return prices

if __name__ == "__main__":
    df = get_data_pipeline()
    print(df.head())
