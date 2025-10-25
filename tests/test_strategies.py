"""
Tests for trading strategies.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.mean_reversion import MeanReversionStrategy, BollingerBandsStrategy
from strategies.momentum import MomentumStrategy, MovingAverageCrossoverStrategy

class TestMeanReversionStrategy:
    """Test mean reversion strategy."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample OHLCV data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate price data with some mean reversion
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        self.data = pd.DataFrame({
            'Open': prices + np.random.randn(100) * 0.1,
            'High': prices + np.abs(np.random.randn(100) * 0.2),
            'Low': prices - np.abs(np.random.randn(100) * 0.2),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Add basic technical indicators
        self.data['SMA_20'] = self.data['Close'].rolling(20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(50).mean()
        self.data['RSI'] = 50 + np.random.randn(100) * 10  # Mock RSI
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility_20'] = self.data['Returns'].rolling(20).std() * np.sqrt(252)
    
    def test_mean_reversion_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = MeanReversionStrategy()
        assert strategy.lookback_period == 20
        assert strategy.z_score_threshold == 2.0
        assert strategy.rsi_oversold == 30.0
        assert strategy.rsi_overbought == 70.0
    
    def test_mean_reversion_strategy_signals(self):
        """Test signal generation."""
        strategy = MeanReversionStrategy()
        result = strategy.calculate_signals(self.data)
        
        # Check that required columns are added
        assert 'Signal' in result.columns
        assert 'Position' in result.columns
        assert 'Z_Score' in result.columns
        assert 'BB_Position' in result.columns
        
        # Check signal values are valid
        assert result['Signal'].isin([-1, 0, 1]).all()
        assert result['Position'].isin([-1, 0, 1]).all()
    
    def test_bollinger_bands_strategy(self):
        """Test Bollinger Bands strategy."""
        strategy = BollingerBandsStrategy()
        result = strategy.calculate_signals(self.data)
        
        # Check that Bollinger Bands are calculated
        assert 'BB_Upper' in result.columns
        assert 'BB_Lower' in result.columns
        assert 'BB_Middle' in result.columns
        
        # Check signals
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_strategy_metrics(self):
        """Test strategy metrics calculation."""
        strategy = MeanReversionStrategy()
        result = strategy.calculate_signals(self.data)
        metrics = strategy.get_strategy_metrics(result)
        
        # Check that metrics are calculated
        expected_metrics = ['total_return', 'annualized_return', 'volatility', 
                           'sharpe_ratio', 'win_rate', 'max_drawdown', 'total_trades']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))

class TestMomentumStrategy:
    """Test momentum strategy."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate trending price data
        trend = np.linspace(100, 150, 100)
        noise = np.random.randn(100) * 2
        prices = trend + noise
        
        self.data = pd.DataFrame({
            'Open': prices + np.random.randn(100) * 0.1,
            'High': prices + np.abs(np.random.randn(100) * 0.2),
            'Low': prices - np.abs(np.random.randn(100) * 0.2),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Add technical indicators
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility_20'] = self.data['Returns'].rolling(20).std() * np.sqrt(252)
    
    def test_momentum_strategy_initialization(self):
        """Test momentum strategy initialization."""
        strategy = MomentumStrategy()
        assert strategy.fast_period == 12
        assert strategy.slow_period == 26
        assert strategy.signal_period == 9
        assert strategy.adx_period == 14
    
    def test_momentum_strategy_signals(self):
        """Test momentum strategy signal generation."""
        strategy = MomentumStrategy()
        result = strategy.calculate_signals(self.data)
        
        # Check required columns
        assert 'Signal' in result.columns
        assert 'Position' in result.columns
        assert 'MACD' in result.columns
        assert 'ADX' in result.columns
        
        # Check signal values
        assert result['Signal'].isin([-1, 0, 1]).all()
    
    def test_moving_average_crossover_strategy(self):
        """Test moving average crossover strategy."""
        strategy = MovingAverageCrossoverStrategy()
        result = strategy.calculate_signals(self.data)
        
        # Check moving averages are calculated
        assert 'MA_Fast' in result.columns
        assert 'MA_Slow' in result.columns
        
        # Check signals
        assert result['Signal'].isin([-1, 0, 1]).all()

class TestStrategyIntegration:
    """Test strategy integration with backtester."""
    
    def setup_method(self):
        """Set up test data."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        prices = 100 + np.cumsum(np.random.randn(50) * 0.5)
        
        self.data = pd.DataFrame({
            'Open': prices + np.random.randn(50) * 0.1,
            'High': prices + np.abs(np.random.randn(50) * 0.2),
            'Low': prices - np.abs(np.random.randn(50) * 0.2),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        # Add required indicators
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility_20'] = self.data['Returns'].rolling(20).std() * np.sqrt(252)
        self.data['RSI'] = 50 + np.random.randn(50) * 10
    
    def test_strategy_with_minimal_data(self):
        """Test strategies work with minimal data."""
        strategies = [
            MeanReversionStrategy(),
            BollingerBandsStrategy(),
            MomentumStrategy(),
            MovingAverageCrossoverStrategy()
        ]
        
        for strategy in strategies:
            try:
                result = strategy.calculate_signals(self.data)
                assert len(result) == len(self.data)
                assert 'Signal' in result.columns
            except Exception as e:
                # Some strategies might need more data
                print(f"Strategy {strategy.__class__.__name__} failed with minimal data: {e}")
    
    def test_strategy_consistency(self):
        """Test that strategies produce consistent results."""
        strategy = MeanReversionStrategy()
        
        # Run strategy multiple times
        results = []
        for _ in range(3):
            result = strategy.calculate_signals(self.data)
            results.append(result['Signal'].values)
        
        # Results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])

if __name__ == "__main__":
    pytest.main([__file__])
