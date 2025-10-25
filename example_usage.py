"""
Example usage of the Quantitative Strategy Simulator

This script demonstrates how to use the backtesting framework
with different strategies and data sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import our modules
from data.download_data import DataManager
from strategies.mean_reversion import MeanReversionStrategy, BollingerBandsStrategy
from strategies.momentum import MomentumStrategy, MovingAverageCrossoverStrategy
from backtester import Backtester, PortfolioBacktester

def main():
    """Main example demonstrating the backtesting framework."""
    
    print("🚀 Quantitative Strategy Simulator - Example Usage")
    print("=" * 60)
    
    # Initialize data manager
    dm = DataManager()
    
    # Download data for multiple tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"📊 Downloading data for {tickers} from {start_date} to {end_date}")
    
    # Download data
    data_dict = dm.download_multiple_tickers(tickers, start_date, end_date)
    
    if not data_dict:
        print("❌ Failed to download data")
        return
    
    # Process data for first ticker
    ticker = list(data_dict.keys())[0]
    data = data_dict[ticker]
    
    print(f"✅ Downloaded {ticker} data: {data.shape}")
    
    # Calculate returns first
    data_with_returns = dm.calculate_returns(data)
    
    # Add technical indicators
    data_with_indicators = dm.add_technical_indicators(data_with_returns)
    
    print(f"📈 Added technical indicators: {data_with_returns.shape}")
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    # Test different strategies
    strategies = [
        MeanReversionStrategy(),
        BollingerBandsStrategy(),
        MomentumStrategy(),
        MovingAverageCrossoverStrategy()
    ]
    
    print(f"\n🧪 Testing {len(strategies)} strategies...")
    
    # Run backtests
    results = []
    for strategy in strategies:
        try:
            print(f"\n📊 Testing {strategy.__class__.__name__}...")
            result = backtester.run_backtest(data_with_returns, strategy)
            results.append(result)
            
            # Print key metrics
            metrics = result['metrics']
            print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
    
    # Compare strategies
    if results:
        print(f"\n📊 Strategy Comparison:")
        print("-" * 80)
        
        comparison_data = []
        for result in results:
            metrics = result['metrics']
            metrics['strategy'] = result['strategy_name']
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison table
        display_cols = ['strategy', 'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        if all(col in comparison_df.columns for col in display_cols):
            print(comparison_df[display_cols].to_string(index=False, float_format='%.3f'))
        
        # Plot results for best strategy
        best_strategy_idx = comparison_df['sharpe_ratio'].idxmax()
        best_result = results[best_strategy_idx]
        
        print(f"\n📈 Plotting results for best strategy: {best_result['strategy_name']}")
        backtester.plot_results(best_result, save_path=f"backtest_results_{ticker}.png")
    
    # Portfolio backtesting example
    print(f"\n🏦 Portfolio Backtesting Example:")
    portfolio_backtester = PortfolioBacktester()
    
    # Test portfolio with mean reversion strategy
    mean_reversion = MeanReversionStrategy()
    portfolio_results = portfolio_backtester.run_portfolio_backtest(data_dict, mean_reversion)
    
    print(f"✅ Portfolio backtest completed for {len(portfolio_results)} assets")
    
    # Print portfolio summary
    portfolio_returns = []
    for ticker, result in portfolio_results.items():
        metrics = result['metrics']
        portfolio_returns.append(metrics.get('total_return', 0))
        print(f"   {ticker}: {metrics.get('total_return', 0):.2%} return")
    
    if portfolio_returns:
        avg_return = np.mean(portfolio_returns)
        print(f"   Portfolio Average Return: {avg_return:.2%}")
    
    print(f"\n🎉 Example completed successfully!")
    print(f"📁 Results saved to current directory")

def quick_demo():
    """Quick demonstration with minimal setup."""
    
    print("⚡ Quick Demo - Mean Reversion Strategy")
    print("=" * 50)
    
    # Initialize components
    dm = DataManager()
    backtester = Backtester(initial_capital=50000)
    strategy = MeanReversionStrategy()
    
    # Download data
    data = dm.download_yfinance_data('AAPL', '2022-01-01', '2023-12-31')
    data_with_returns = dm.calculate_returns(data)
    data_with_indicators = dm.add_technical_indicators(data_with_returns)
    
    # Run backtest
    result = backtester.run_backtest(data_with_returns, strategy)
    
    # Display results
    metrics = result['metrics']
    print(f"📊 AAPL Mean Reversion Strategy Results:")
    print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_demo()
    else:
        main()
