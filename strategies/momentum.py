"""
Momentum Trading Strategy

This strategy identifies trends and follows them. Common indicators include
moving average crossovers, MACD, and trend strength indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MomentumStrategy:
    """
    Momentum strategy using multiple trend-following indicators.
    
    This strategy looks for:
    1. Moving average crossovers
    2. MACD signals
    3. Trend strength (ADX)
    4. Volume confirmation
    """
    
    def __init__(self, 
                 fast_period: int = 12,
                 slow_period: int = 26,
                 signal_period: int = 9,
                 adx_period: int = 14,
                 volume_threshold: float = 1.2):
        """
        Initialize momentum strategy.
        
        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            signal_period: MACD signal line period
            adx_period: ADX calculation period
            volume_threshold: Volume multiplier for confirmation
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.adx_period = adx_period
        self.volume_threshold = volume_threshold
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on momentum indicators.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with signals and positions
        """
        df = data.copy()
        
        # Calculate additional momentum indicators
        df = self._calculate_momentum_indicators(df)
        
        # Generate signals
        df = self._generate_signals(df)
        
        # Calculate position sizes
        df = self._calculate_position_sizes(df)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-specific indicators."""
        
        # MACD
        df['EMA_Fast'] = df['Close'].ewm(span=self.fast_period).mean()
        df['EMA_Slow'] = df['Close'].ewm(span=self.slow_period).mean()
        df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
        df['MACD_Signal'] = df['MACD'].ewm(span=self.signal_period).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # ADX (Average Directional Index)
        df = self._calculate_adx(df)
        
        # Rate of Change
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        
        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        df['DM_Plus'] = np.where(
            (high - high.shift(1)) > (low.shift(1) - low),
            np.maximum(high - high.shift(1), 0),
            0
        )
        df['DM_Minus'] = np.where(
            (low.shift(1) - low) > (high - high.shift(1)),
            np.maximum(low.shift(1) - low, 0),
            0
        )
        
        # Smoothed values
        df['TR_Smooth'] = df['TR'].rolling(window=self.adx_period).mean()
        df['DM_Plus_Smooth'] = df['DM_Plus'].rolling(window=self.adx_period).mean()
        df['DM_Minus_Smooth'] = df['DM_Minus'].rolling(window=self.adx_period).mean()
        
        # Directional Indicators
        df['DI_Plus'] = 100 * (df['DM_Plus_Smooth'] / df['TR_Smooth'])
        df['DI_Minus'] = 100 * (df['DM_Minus_Smooth'] / df['TR_Smooth'])
        
        # ADX
        df['DX'] = 100 * abs(df['DI_Plus'] - df['DI_Minus']) / (df['DI_Plus'] + df['DI_Minus'])
        df['ADX'] = df['DX'].rolling(window=self.adx_period).mean()
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate buy/sell signals based on momentum logic."""
        
        # Initialize signal columns
        df['Signal'] = 0
        df['Position'] = 0
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        
        position = 0
        entry_price = 0
        
        for i in range(max(self.slow_period, self.adx_period), len(df)):
            current_price = df['Close'].iloc[i]
            
            # Get indicator values
            macd = df['MACD'].iloc[i]
            macd_signal = df['MACD_Signal'].iloc[i]
            macd_histogram = df['MACD_Histogram'].iloc[i]
            adx = df['ADX'].iloc[i]
            di_plus = df['DI_Plus'].iloc[i]
            di_minus = df['DI_Minus'].iloc[i]
            roc = df['ROC'].iloc[i]
            volume_ratio = df['Volume_Ratio'].iloc[i]
            
            # Long entry conditions (strong uptrend)
            if (position == 0 and 
                macd > macd_signal and  # MACD bullish crossover
                macd_histogram > 0 and  # MACD histogram positive
                adx > 25 and  # Strong trend
                di_plus > di_minus and  # Upward direction
                roc > 0 and  # Positive momentum
                volume_ratio > self.volume_threshold):  # Volume confirmation
                
                df.iloc[i, df.columns.get_loc('Signal')] = 1  # Buy signal
                df.iloc[i, df.columns.get_loc('Position')] = 1
                df.iloc[i, df.columns.get_loc('Entry_Price')] = current_price
                position = 1
                entry_price = current_price
                
            # Short entry conditions (strong downtrend)
            elif (position == 0 and 
                  macd < macd_signal and  # MACD bearish crossover
                  macd_histogram < 0 and  # MACD histogram negative
                  adx > 25 and  # Strong trend
                  di_minus > di_plus and  # Downward direction
                  roc < 0 and  # Negative momentum
                  volume_ratio > self.volume_threshold):  # Volume confirmation
                
                df.iloc[i, df.columns.get_loc('Signal')] = -1  # Sell signal
                df.iloc[i, df.columns.get_loc('Position')] = -1
                df.iloc[i, df.columns.get_loc('Entry_Price')] = current_price
                position = -1
                entry_price = current_price
                
            # Exit conditions
            elif position != 0:
                # Long exit conditions
                if (position == 1 and 
                    (macd < macd_signal or  # MACD bearish crossover
                     adx < 20 or  # Weak trend
                     di_minus > di_plus)):  # Direction change
                    
                    df.iloc[i, df.columns.get_loc('Signal')] = -1  # Exit long
                    df.iloc[i, df.columns.get_loc('Position')] = 0
                    df.iloc[i, df.columns.get_loc('Exit_Price')] = current_price
                    position = 0
                    
                # Short exit conditions
                elif (position == -1 and 
                      (macd > macd_signal or  # MACD bullish crossover
                       adx < 20 or  # Weak trend
                       di_plus > di_minus)):  # Direction change
                    
                    df.iloc[i, df.columns.get_loc('Signal')] = 1  # Exit short
                    df.iloc[i, df.columns.get_loc('Position')] = 0
                    df.iloc[i, df.columns.get_loc('Exit_Price')] = current_price
                    position = 0
                
                # Update current position
                df.iloc[i, df.columns.get_loc('Position')] = position
        
        return df
    
    def _calculate_position_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes based on trend strength and momentum."""
        df['Position_Size'] = 0.0
        
        for i in range(len(df)):
            if df['Position'].iloc[i] != 0:
                # Base position size
                base_size = 0.1  # 10% of portfolio
                
                # Adjust based on ADX (stronger trend = larger position)
                adx = df['ADX'].iloc[i]
                trend_strength_multiplier = min(adx / 50.0, 2.0) if not pd.isna(adx) else 1.0
                
                # Adjust based on momentum (stronger momentum = larger position)
                momentum = abs(df['Momentum'].iloc[i])
                momentum_multiplier = min(momentum / (df['Close'].iloc[i] * 0.05), 2.0) if not pd.isna(momentum) else 1.0
                
                # Final position size
                df.iloc[i, df.columns.get_loc('Position_Size')] = (
                    base_size * trend_strength_multiplier * momentum_multiplier
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

class MovingAverageCrossoverStrategy(MomentumStrategy):
    """
    Simplified momentum strategy using moving average crossovers.
    """
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate signals using moving average crossovers only."""
        df = data.copy()
        
        # Calculate moving averages
        df['MA_Fast'] = df['Close'].rolling(window=self.fast_period).mean()
        df['MA_Slow'] = df['Close'].rolling(window=self.slow_period).mean()
        
        # Generate signals
        df['Signal'] = 0
        df['Position'] = 0
        
        for i in range(self.slow_period, len(df)):
            ma_fast = df['MA_Fast'].iloc[i]
            ma_slow = df['MA_Slow'].iloc[i]
            ma_fast_prev = df['MA_Fast'].iloc[i-1]
            ma_slow_prev = df['MA_Slow'].iloc[i-1]
            
            # Golden cross (bullish)
            if (ma_fast > ma_slow and ma_fast_prev <= ma_slow_prev):
                df.iloc[i, df.columns.get_loc('Signal')] = 1
                df.iloc[i, df.columns.get_loc('Position')] = 1
            # Death cross (bearish)
            elif (ma_fast < ma_slow and ma_fast_prev >= ma_slow_prev):
                df.iloc[i, df.columns.get_loc('Signal')] = -1
                df.iloc[i, df.columns.get_loc('Position')] = 0
        
        return df
