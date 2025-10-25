"""
Mean Reversion Trading Strategy

This strategy identifies when a stock has deviated significantly from its mean
and expects it to revert back. Common indicators include Bollinger Bands,
Z-scores, and RSI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MeanReversionStrategy:
    """
    Mean reversion strategy using multiple indicators.
    
    This strategy looks for:
    1. Price deviation from moving average (Z-score)
    2. Bollinger Band position
    3. RSI oversold/overbought conditions
    4. Volume confirmation
    """
    
    def __init__(self, 
                 lookback_period: int = 20,
                 z_score_threshold: float = 2.0,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 volume_threshold: float = 1.5):
        """
        Initialize mean reversion strategy.
        
        Args:
            lookback_period: Period for moving average calculation
            z_score_threshold: Z-score threshold for entry signals
            rsi_oversold: RSI level considered oversold
            rsi_overbought: RSI level considered overbought
            volume_threshold: Volume multiplier for confirmation
        """
        self.lookback_period = lookback_period
        self.z_score_threshold = z_score_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold = volume_threshold
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on mean reversion indicators.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with signals and positions
        """
        df = data.copy()
        
        # Calculate Z-score (price deviation from mean)
        df['Price_MA'] = df['Close'].rolling(window=self.lookback_period).mean()
        df['Price_Std'] = df['Close'].rolling(window=self.lookback_period).std()
        df['Z_Score'] = (df['Close'] - df['Price_MA']) / df['Price_Std']
        
        # Calculate Bollinger Bands if not present
        if 'BB_Upper' not in df.columns or 'BB_Lower' not in df.columns:
            bb_window = min(20, len(df))
            df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
            bb_std = df['Close'].rolling(window=bb_window).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Bollinger Band position
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Calculate RSI if not present
        if 'RSI' not in df.columns:
            df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Calculate volatility if not present
        if 'Volatility_20' not in df.columns:
            vol_window = min(20, len(df))
            df['Volatility_20'] = df['Returns'].rolling(window=vol_window).std() * np.sqrt(252)
        
        # Volume ratio (current vs average)
        df['Volume_MA'] = df['Volume'].rolling(window=self.lookback_period).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Generate signals
        df = self._generate_signals(df)
        
        # Calculate position sizes
        df = self._calculate_position_sizes(df)
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on mean reversion logic."""
        
        # Initialize signal columns
        df['Signal'] = 0
        df['Position'] = 0
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        
        position = 0
        entry_price = 0
        
        for i in range(self.lookback_period, len(df)):
            current_price = df['Close'].iloc[i]
            z_score = df['Z_Score'].iloc[i]
            rsi = df['RSI'].iloc[i]
            bb_position = df['BB_Position'].iloc[i]
            volume_ratio = df['Volume_Ratio'].iloc[i]
            
            # Long entry conditions (oversold)
            if (position == 0 and 
                z_score < -self.z_score_threshold and
                rsi < self.rsi_oversold and
                bb_position < 0.2 and
                volume_ratio > self.volume_threshold):
                
                df.iloc[i, df.columns.get_loc('Signal')] = 1  # Buy signal
                df.iloc[i, df.columns.get_loc('Position')] = 1
                df.iloc[i, df.columns.get_loc('Entry_Price')] = current_price
                position = 1
                entry_price = current_price
                
            # Short entry conditions (overbought)
            elif (position == 0 and 
                  z_score > self.z_score_threshold and
                  rsi > self.rsi_overbought and
                  bb_position > 0.8 and
                  volume_ratio > self.volume_threshold):
                
                df.iloc[i, df.columns.get_loc('Signal')] = -1  # Sell signal
                df.iloc[i, df.columns.get_loc('Position')] = -1
                df.iloc[i, df.columns.get_loc('Entry_Price')] = current_price
                position = -1
                entry_price = current_price
                
            # Exit conditions
            elif position != 0:
                # Long exit conditions
                if (position == 1 and 
                    (z_score > 0 or  # Price back to mean
                     rsi > 50 or      # RSI neutral
                     bb_position > 0.5)):  # Back to middle of BB
                    
                    df.iloc[i, df.columns.get_loc('Signal')] = -1  # Exit long
                    df.iloc[i, df.columns.get_loc('Position')] = 0
                    df.iloc[i, df.columns.get_loc('Exit_Price')] = current_price
                    position = 0
                    
                # Short exit conditions
                elif (position == -1 and 
                      (z_score < 0 or  # Price back to mean
                       rsi < 50 or     # RSI neutral
                       bb_position < 0.5)):  # Back to middle of BB
                    
                    df.iloc[i, df.columns.get_loc('Signal')] = 1  # Exit short
                    df.iloc[i, df.columns.get_loc('Position')] = 0
                    df.iloc[i, df.columns.get_loc('Exit_Price')] = current_price
                    position = 0
                
                # Update current position
                df.iloc[i, df.columns.get_loc('Position')] = position
        
        return df
    
    def _calculate_position_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes based on volatility and confidence."""
        df['Position_Size'] = 0.0
        
        for i in range(len(df)):
            if df['Position'].iloc[i] != 0:
                # Base position size (can be adjusted based on confidence)
                base_size = 0.1  # 10% of portfolio
                
                # Adjust based on Z-score magnitude (stronger signal = larger position)
                z_score_magnitude = abs(df['Z_Score'].iloc[i])
                confidence_multiplier = min(z_score_magnitude / self.z_score_threshold, 2.0)
                
                # Adjust based on volatility (lower volatility = larger position)
                volatility = df['Volatility_20'].iloc[i]
                volatility_multiplier = 1.0 / (1.0 + volatility) if not pd.isna(volatility) else 1.0
                
                # Final position size
                df.iloc[i, df.columns.get_loc('Position_Size')] = (
                    base_size * confidence_multiplier * volatility_multiplier
                )
        
        return df
    
    def get_strategy_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        signals = df['Signal'].dropna()
        returns = df['Returns'].dropna()
        
        if len(signals) == 0:
            return {}
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * returns
        
        # Basic metrics
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (strategy_returns != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades
        }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class BollingerBandsStrategy(MeanReversionStrategy):
    """
    Simplified mean reversion strategy using only Bollinger Bands.
    """
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        super().__init__()
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate signals using Bollinger Bands only."""
        df = data.copy()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=self.bb_period).mean()
        bb_std = df['Close'].rolling(window=self.bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * self.bb_std)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * self.bb_std)
        
        # Generate signals
        df['Signal'] = 0
        df['Position'] = 0
        
        for i in range(self.bb_period, len(df)):
            current_price = df['Close'].iloc[i]
            bb_upper = df['BB_Upper'].iloc[i]
            bb_lower = df['BB_Lower'].iloc[i]
            
            # Buy when price touches lower band
            if current_price <= bb_lower:
                df.iloc[i, df.columns.get_loc('Signal')] = 1
                df.iloc[i, df.columns.get_loc('Position')] = 1
            # Sell when price touches upper band
            elif current_price >= bb_upper:
                df.iloc[i, df.columns.get_loc('Signal')] = -1
                df.iloc[i, df.columns.get_loc('Position')] = 0
        
        return df
