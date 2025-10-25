"""
Comprehensive Backtesting Engine for Quantitative Trading Strategies

This module provides a robust backtesting framework with:
- Multiple strategy support
- Risk management
- Performance metrics
- Visualization
- Portfolio optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class Backtester:
    """
    Comprehensive backtesting engine for trading strategies.
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction)
            slippage: Slippage per trade (as fraction)
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> Dict:
        """
        Run backtest for a given strategy.
        
        Args:
            data: OHLCV data with technical indicators
            strategy: Strategy object with calculate_signals method
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
            
        Returns:
            Dictionary with backtest results
        """
        # Filter data by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if len(data) == 0:
            raise ValueError("No data available for the specified date range")
        
        # Calculate strategy signals
        data_with_signals = strategy.calculate_signals(data)
        
        # Run backtest simulation
        results = self._simulate_trading(data_with_signals)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(results)
        
        # Add strategy-specific metrics
        strategy_metrics = strategy.get_strategy_metrics(data_with_signals)
        metrics.update(strategy_metrics)
        
        return {
            'data': data_with_signals,
            'results': results,
            'metrics': metrics,
            'strategy_name': strategy.__class__.__name__
        }
    
    def _simulate_trading(self, data: pd.DataFrame) -> pd.DataFrame:
        """Simulate trading based on signals."""
        df = data.copy()
        
        # Initialize portfolio tracking
        df['Portfolio_Value'] = self.initial_capital
        df['Cash'] = self.initial_capital
        df['Shares'] = 0.0
        df['Trade_Value'] = 0.0
        df['Cumulative_Returns'] = 0.0
        df['Drawdown'] = 0.0
        
        cash = self.initial_capital
        shares = 0.0
        portfolio_value = self.initial_capital
        peak_value = self.initial_capital
        
        for i in range(len(df)):
            current_price = df['Close'].iloc[i]
            signal = df['Signal'].iloc[i]
            position_size = df.get('Position_Size', pd.Series([0.1] * len(df))).iloc[i]
            
            # Calculate trade value based on position size
            if signal != 0:
                trade_value = portfolio_value * position_size
            else:
                trade_value = 0
            
            # Execute trades
            if signal == 1 and cash > 0:  # Buy signal
                # Calculate shares to buy
                shares_to_buy = trade_value / (current_price * (1 + self.slippage))
                cost = shares_to_buy * current_price * (1 + self.slippage + self.commission)
                
                if cost <= cash:
                    shares += shares_to_buy
                    cash -= cost
                    
            elif signal == -1 and shares > 0:  # Sell signal
                # Calculate shares to sell
                shares_to_sell = shares
                proceeds = shares_to_sell * current_price * (1 - self.slippage - self.commission)
                
                shares = 0
                cash += proceeds
            
            # Update portfolio value
            portfolio_value = cash + shares * current_price
            
            # Track peak for drawdown calculation
            if portfolio_value > peak_value:
                peak_value = portfolio_value
            
            # Calculate drawdown
            drawdown = (portfolio_value - peak_value) / peak_value
            
            # Update DataFrame
            df.iloc[i, df.columns.get_loc('Cash')] = cash
            df.iloc[i, df.columns.get_loc('Shares')] = shares
            df.iloc[i, df.columns.get_loc('Portfolio_Value')] = portfolio_value
            df.iloc[i, df.columns.get_loc('Trade_Value')] = trade_value
            df.iloc[i, df.columns.get_loc('Cumulative_Returns')] = (portfolio_value / self.initial_capital) - 1
            df.iloc[i, df.columns.get_loc('Drawdown')] = drawdown
        
        return df
    
    def _calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        portfolio_values = results['Portfolio_Value']
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / self.initial_capital) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Drawdown metrics
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate and trade analysis
        trade_returns = returns[returns != 0]
        if len(trade_returns) > 0:
            win_rate = (trade_returns > 0).sum() / len(trade_returns)
            avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).sum() > 0 else 0
            avg_loss = trade_returns[trade_returns < 0].mean() if (trade_returns < 0).sum() > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Additional metrics
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trade_returns)
        }
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        
        return (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    def compare_strategies(self, 
                         data: pd.DataFrame, 
                         strategies: List,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            data: OHLCV data
            strategies: List of strategy objects
            start_date: Start date for comparison
            end_date: End date for comparison
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for strategy in strategies:
            try:
                backtest_result = self.run_backtest(data, strategy, start_date, end_date)
                metrics = backtest_result['metrics']
                metrics['strategy'] = strategy.__class__.__name__
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error backtesting {strategy.__class__.__name__}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def plot_results(self, 
                    backtest_result: Dict, 
                    save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive backtest results.
        
        Args:
            backtest_result: Result from run_backtest
            save_path: Path to save plots (optional)
        """
        data = backtest_result['data']
        results = backtest_result['results']
        metrics = backtest_result['metrics']
        strategy_name = backtest_result['strategy_name']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{strategy_name} - Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Price and Signals
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['Close'], label='Price', alpha=0.7)
        
        # Plot buy signals
        buy_signals = data[data['Signal'] == 1]
        if len(buy_signals) > 0:
            ax1.scatter(buy_signals.index, buy_signals['Close'], 
                       color='green', marker='^', s=100, label='Buy Signal', zorder=5)
        
        # Plot sell signals
        sell_signals = data[data['Signal'] == -1]
        if len(sell_signals) > 0:
            ax1.scatter(sell_signals.index, sell_signals['Close'], 
                       color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Price and Trading Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Portfolio Value
        ax2 = axes[0, 1]
        ax2.plot(results.index, results['Portfolio_Value'], label='Portfolio Value', linewidth=2)
        ax2.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative Returns
        ax3 = axes[1, 0]
        ax3.plot(results.index, results['Cumulative_Returns'] * 100, 
                label='Strategy Returns', linewidth=2)
        
        # Benchmark (buy and hold)
        benchmark_returns = (data['Close'] / data['Close'].iloc[0] - 1) * 100
        ax3.plot(data.index, benchmark_returns, 
                label='Buy & Hold', alpha=0.7, linestyle='--')
        
        ax3.set_title('Cumulative Returns Comparison')
        ax3.set_ylabel('Cumulative Returns (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown
        ax4 = axes[1, 1]
        ax4.fill_between(results.index, results['Drawdown'] * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax4.set_title('Drawdown Over Time')
        ax4.set_ylabel('Drawdown (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print metrics
        self._print_metrics(metrics, strategy_name)
    
    def _print_metrics(self, metrics: Dict[str, float], strategy_name: str) -> None:
        """Print formatted performance metrics."""
        print(f"\n{'='*50}")
        print(f"STRATEGY: {strategy_name}")
        print(f"{'='*50}")
        
        print(f"Total Return:        {metrics.get('total_return', 0):.2%}")
        print(f"Annualized Return:   {metrics.get('annualized_return', 0):.2%}")
        print(f"Volatility:          {metrics.get('volatility', 0):.2%}")
        print(f"Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Sortino Ratio:       {metrics.get('sortino_ratio', 0):.3f}")
        print(f"Calmar Ratio:        {metrics.get('calmar_ratio', 0):.3f}")
        print(f"Max Drawdown:        {metrics.get('max_drawdown', 0):.2%}")
        print(f"Win Rate:            {metrics.get('win_rate', 0):.2%}")
        print(f"Profit Factor:       {metrics.get('profit_factor', 0):.3f}")
        print(f"Total Trades:        {metrics.get('total_trades', 0)}")
        
        print(f"{'='*50}")

class PortfolioBacktester(Backtester):
    """
    Extended backtester for portfolio-level strategies.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.portfolio_weights = {}
    
    def run_portfolio_backtest(self, 
                              data_dict: Dict[str, pd.DataFrame], 
                              strategy,
                              rebalance_frequency: str = 'monthly') -> Dict:
        """
        Run backtest for portfolio of assets.
        
        Args:
            data_dict: Dictionary mapping ticker to DataFrame
            strategy: Strategy object
            rebalance_frequency: How often to rebalance portfolio
            
        Returns:
            Portfolio backtest results
        """
        # This would implement portfolio-level backtesting
        # For now, return individual asset results
        results = {}
        
        for ticker, data in data_dict.items():
            try:
                result = self.run_backtest(data, strategy)
                results[ticker] = result
            except Exception as e:
                logger.error(f"Error backtesting {ticker}: {e}")
                continue
        
        return results
