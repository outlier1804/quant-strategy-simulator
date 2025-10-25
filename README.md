# 🚀 Quantitative Strategy Simulator

A comprehensive Python framework for backtesting and evaluating algorithmic trading strategies. This project demonstrates advanced quantitative finance skills including data engineering, strategy development, risk management, and performance analysis.

## ✨ Features

### 📊 **Data Management**
- **Multi-source data ingestion** (Yahoo Finance, Alpha Vantage, custom APIs)
- **Intelligent caching** with automatic data validation
- **Technical indicators** (RSI, MACD, Bollinger Bands, ADX, etc.)
- **Data cleaning** and preprocessing pipelines

### 🎯 **Trading Strategies**
- **Mean Reversion Strategies**
  - Z-score based entry/exit signals
  - Bollinger Bands mean reversion
  - RSI oversold/overbought detection
  - Volume confirmation filters

- **Momentum Strategies**
  - MACD trend following
  - Moving average crossovers
  - ADX trend strength analysis
  - Rate of change momentum

### 🔬 **Advanced Backtesting**
- **Realistic simulation** with commissions and slippage
- **Risk management** with position sizing
- **Performance metrics** (Sharpe, Sortino, Calmar ratios)
- **Drawdown analysis** and risk assessment
- **Strategy comparison** and optimization

### 📈 **Visualization & Analysis**
- **Interactive charts** with trading signals
- **Performance dashboards** with key metrics
- **Drawdown visualization** and risk analysis
- **Strategy comparison** tables and plots

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/outlier1804/quant-strategy-simulator.git
cd quant-strategy-simulator

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/
```

## 🚀 Quick Start

### Basic Usage

```python
from data.download_data import DataManager
from strategies.mean_reversion import MeanReversionStrategy
from backtester import Backtester

# Download data
dm = DataManager()
data = dm.download_yfinance_data('AAPL', '2020-01-01', '2023-12-31')
data_with_indicators = dm.add_technical_indicators(data)
data_with_returns = dm.calculate_returns(data_with_indicators)

# Initialize strategy and backtester
strategy = MeanReversionStrategy()
backtester = Backtester(initial_capital=100000)

# Run backtest
result = backtester.run_backtest(data_with_returns, strategy)

# Plot results
backtester.plot_results(result)
```

### Advanced Usage

```python
# Compare multiple strategies
strategies = [
    MeanReversionStrategy(),
    MomentumStrategy(),
    BollingerBandsStrategy()
]

comparison_df = backtester.compare_strategies(data_with_returns, strategies)
print(comparison_df)

# Portfolio backtesting
portfolio_backtester = PortfolioBacktester()
tickers = ['AAPL', 'MSFT', 'GOOGL']
data_dict = dm.download_multiple_tickers(tickers, '2020-01-01', '2023-12-31')
portfolio_results = portfolio_backtester.run_portfolio_backtest(data_dict, strategy)
```

## 📊 Example Results

### Mean Reversion Strategy Performance
```
==================================================
STRATEGY: MeanReversionStrategy
==================================================
Total Return:        15.23%
Annualized Return:   4.87%
Volatility:          12.45%
Sharpe Ratio:        0.392
Sortino Ratio:       0.456
Calmar Ratio:        0.234
Max Drawdown:        -8.92%
Win Rate:            58.33%
Profit Factor:       1.234
Total Trades:        24
==================================================
```

## 🏗️ Architecture

```
quant-strategy-simulator/
├── data/
│   └── download_data.py          # Data ingestion and management
├── strategies/
│   ├── mean_reversion.py         # Mean reversion strategies
│   └── momentum.py               # Momentum strategies
├── backtester.py                 # Core backtesting engine
├── example_usage.py              # Usage examples
├── tests/                        # Comprehensive test suite
└── requirements.txt              # Dependencies
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_strategies.py -v
```

## 📈 Performance Metrics

The framework calculates comprehensive performance metrics:

- **Return Metrics**: Total return, annualized return, volatility
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown
- **Trade Analysis**: Win rate, profit factor, average win/loss
- **Risk Management**: Position sizing, drawdown analysis

## 🔧 Customization

### Adding New Strategies

```python
class CustomStrategy:
    def __init__(self, param1=10, param2=20):
        self.param1 = param1
        self.param2 = param2
    
    def calculate_signals(self, data):
        # Implement your strategy logic
        df = data.copy()
        df['Signal'] = 0  # Your signal logic here
        return df
    
    def get_strategy_metrics(self, data):
        # Calculate strategy-specific metrics
        return {'custom_metric': 0.5}
```

### Custom Data Sources

```python
class CustomDataManager(DataManager):
    def download_custom_data(self, source, **kwargs):
        # Implement custom data source
        pass
```

## 🎯 Use Cases

This framework is perfect for:

- **Quantitative Research**: Testing new trading ideas
- **Strategy Development**: Building and optimizing algorithms
- **Risk Management**: Analyzing portfolio risk and drawdowns
- **Performance Analysis**: Comparing strategy effectiveness
- **Educational**: Learning quantitative finance concepts

## 🚀 Advanced Features

### Portfolio Optimization
- Multi-asset backtesting
- Portfolio rebalancing
- Risk budgeting
- Correlation analysis

### Machine Learning Integration
- Feature engineering for ML models
- Strategy parameter optimization
- Ensemble methods
- Model validation

### Real-time Trading
- Live data feeds
- Real-time signal generation
- Order execution simulation
- Risk monitoring

## 📚 Documentation

- [Strategy Development Guide](docs/strategy_development.md)
- [API Reference](docs/api_reference.md)
- [Performance Metrics](docs/performance_metrics.md)
- [Testing Guide](docs/testing.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with Python, Pandas, NumPy, and Matplotlib
- Inspired by professional quantitative trading frameworks
- Designed for educational and research purposes

## 📞 Contact

For questions or collaboration opportunities:
- GitHub: [@outlier1804](https://github.com/outlier1804)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

**⭐ Star this repository if you find it helpful!**
