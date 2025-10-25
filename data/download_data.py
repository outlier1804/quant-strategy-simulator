import yfinance as yf

def download_data(ticker, start_date, end_date, filepath):
    """Downloads historical stock data from Yahoo Finance and saves it to a CSV file."""
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(filepath)
    print(f"Downloaded {ticker} data to {filepath}")

if __name__ == '__main__':
    # Example usage
    download_data('AAPL', '2020-01-01', '2023-01-01', 'data/AAPL.csv')
