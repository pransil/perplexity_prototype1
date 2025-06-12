# load_stock_data.py
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def load_stock_data():
    """
    Loads and processes Tesla stock data to create sequential input-output pairs for time series forecasting.
    
    Returns:
        tuple: (open_prices, close_prices, X, y)
    """
    # Read data and ensure chronological order
    data = pd.read_csv('../stock_data/tesla.csv')
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date')
    
    # Select only numerical features and verify required columns
    numeric_data = data.select_dtypes(include=[np.number])
    required_columns = {'open', 'close'}
    if not required_columns.issubset(numeric_data.columns):
        raise ValueError("CSV missing required columns: 'open' and/or 'close'")
    
    # Extract base series
    open_prices = numeric_data['open'].values
    close_prices = numeric_data['close'].values
    
    # Create sliding window features
    WINDOW_SIZE = 10
    features = numeric_data.values
    
    if len(features) < WINDOW_SIZE:
        raise ValueError(f"Need at least {WINDOW_SIZE} data points for windowing")
    
    # Create sliding windows using stride tricks for efficiency
    X_windows = sliding_window_view(features, WINDOW_SIZE, axis=0)
    X = X_windows.reshape(X_windows.shape[0], -1)  # Flatten window dimensions
    
    # Create targets (next day's closing price)
    y = close_prices[WINDOW_SIZE:]
    
    return open_prices, close_prices, X, y
