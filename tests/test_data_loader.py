import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from data_loader import load_stock_data

def create_test_csv(path, rows=20):
    # Minimal valid CSV for testing
    df = pd.DataFrame({
        'Date': pd.date_range('2020-01-01', periods=rows),
        'open': np.arange(1, rows+1, dtype=float),
        'close': np.arange(101, 101+rows, dtype=float),
        'volume': np.random.randint(1e6, 1e7, size=rows)
    })
    df.to_csv(path, index=False)

def test_load_stock_data_valid(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        test_csv = os.path.join(tmpdir, 'tesla.csv')
        create_test_csv(test_csv, rows=20)
        # Patch the path in the loader
        monkeypatch.setattr('data_loader.DATA_PATH', test_csv, raising=False)
        open_prices, close_prices, X, y = load_stock_data(test_csv)
        assert len(open_prices) == 20
        assert len(close_prices) == 20
        assert X.shape[0] == 11  # 20 - 10 + 1
        assert X.shape[1] == 10 * 3  # 3 features * 10
        assert y.shape[0] == 10  # 20 - 10

def test_missing_columns(tmp_path):
    df = pd.DataFrame({'open': [1,2,3], 'volume': [1,2,3]})
    file = tmp_path / "tesla.csv"
    df.to_csv(file, index=False)
    with pytest.raises(ValueError):
        load_stock_data(str(file))

def test_too_short(tmp_path):
    df = pd.DataFrame({'open': [1,2], 'close': [1,2]})
    file = tmp_path / "tesla.csv"
    df.to_csv(file, index=False)
    with pytest.raises(ValueError):
        load_stock_data(str(file))

def test_nan_handling(tmp_path):
    df = pd.DataFrame({'open': [1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                       'close': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
    file = tmp_path / "tesla.csv"
    df.to_csv(file, index=False)
    # Should not raise, but may produce NaNs in output
    open_prices, close_prices, X, y = load_stock_data(str(file))
    assert np.isnan(open_prices).sum() == 1
    assert np.isnan(X).any()

