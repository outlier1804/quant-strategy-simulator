"""
Data ingestion module for quantitative strategy simulator.
Supports multiple data sources with robust error handling and caching.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """Manages data ingestion from multiple sources with caching and error handling."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def download_yfinance_data(self, 
                              ticker: str, 
                              start_date: str, 
                              end_date: str,
                              interval: str = "1d") -> pd.DataFrame:
        """
        Download data from Yahoo Finance with error handling and caching.
        
        Args:
            ticker: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}_{interval}.csv"
        
        # Check cache first
        if cache_file.exists():
            logger.info(f"Loading {ticker} data from cache")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        try:
            logger.info(f"Downloading {ticker} data from Yahoo Finance")
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {ticker}")
            
            # Handle yfinance data structure
            if isinstance(data.columns, pd.MultiIndex):
                # Handle the case where we have a MultiIndex with ('Price', 'Ticker')
                if len(data.columns.levels) == 2:
                    # Get the price level (first level) and ticker level (second level)
                    price_level = data.columns.get_level_values(0)
                    ticker_level = data.columns.get_level_values(1)
                    
                    # If all tickers are the same, use the price level as column names
                    if len(set(ticker_level)) == 1:
                        data.columns = price_level
                    else:
                        # For multiple tickers, keep the structure but flatten
                        data.columns = [f"{price}_{ticker}" for price, ticker in data.columns]
                else:
                    # For other MultiIndex cases, flatten the columns
                    data.columns = data.columns.get_level_values(1)
            elif len(data.columns) == 5 and all(col == ticker for col in data.columns):
                # Handle case where all columns have the same name
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Clean and validate data
            data = self._clean_data(data)
            
            # Cache the data
            data.to_csv(cache_file)
            logger.info(f"Cached {ticker} data to {cache_file}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            raise
    
    def download_multiple_tickers(self, 
                                 tickers: List[str], 
                                 start_date: str, 
                                 end_date: str,
                                 interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data_dict = {}
        
        for ticker in tickers:
            try:
                data_dict[ticker] = self.download_yfinance_data(ticker, start_date, end_date, interval)
            except Exception as e:
                logger.warning(f"Skipping {ticker}: {e}")
                continue
        
        return data_dict
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data."""
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns
            data.columns = data.columns.get_level_values(1)
        
        # Remove any rows with NaN values
        data = data.dropna()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Found: {data.columns.tolist()}")
        
        # Convert to numeric types
        for col in required_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN values after conversion
        data = data.dropna()
        
        # Remove any rows with zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            data = data[data[col] > 0]
        
        # Remove any rows with negative volume
        data = data[data['Volume'] >= 0]
        
        return data
    
    def get_sp500_tickers(self) -> List[str]:
        """Get list of S&P 500 tickers for portfolio analysis."""
        try:
            # Try to get from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            return sp500_table['Symbol'].tolist()
        except Exception as e:
            logger.warning(f"Failed to fetch S&P 500 list: {e}")
            # Fallback to major tickers
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B', 'UNH', 'JNJ']
    
    def calculate_returns(self, data: pd.DataFrame, method: str = "log") -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            data: OHLCV DataFrame
            method: 'log' for log returns, 'simple' for simple returns
            
        Returns:
            DataFrame with returns
        """
        returns = data.copy()
        
        if method == "log":
            returns['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        else:  # simple returns
            returns['Returns'] = data['Close'].pct_change()
        
        returns['Cumulative_Returns'] = (1 + returns['Returns']).cumprod() - 1
        
        return returns
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to the data."""
        df = data.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        bb_window = min(20, len(df))
        df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
        bb_std = df['Close'].rolling(window=bb_window).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volatility
        vol_window = min(20, len(df))
        df['Volatility_20'] = df['Returns'].rolling(window=vol_window).std() * np.sqrt(252)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

def download_data(ticker: str, start_date: str, end_date: str, filepath: str):
    """
    Legacy function for backward compatibility.
    Downloads historical stock data and saves to CSV.
    """
    dm = DataManager()
    data = dm.download_yfinance_data(ticker, start_date, end_date)
    data.to_csv(filepath)
    print(f"Downloaded {ticker} data to {filepath}")

if __name__ == '__main__':
    # Example usage
    dm = DataManager()
    
    # Download single ticker
    data = dm.download_yfinance_data('AAPL', '2020-01-01', '2023-01-01')
    print(f"Downloaded AAPL data: {data.shape}")
    
    # Add technical indicators
    data_with_indicators = dm.add_technical_indicators(data)
    print(f"Added technical indicators: {data_with_indicators.shape}")
    
    # Download multiple tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    multi_data = dm.download_multiple_tickers(tickers, '2022-01-01', '2023-01-01')
    print(f"Downloaded {len(multi_data)} tickers")
