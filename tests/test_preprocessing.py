import pytest
import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_data_file_exists():
    """Test if raw data exists."""
    assert os.path.exists("dataraw/dataset.csv")

def test_preprocessing_creates_files():
    """Test if preprocessing creates output files."""
    from preprocessing import preprocess_data
    
    preprocess_data(
        input_path="dataraw/dataset.csv",
        output_dir="dataprocessed"
    )
    
    assert os.path.exists("dataprocessed/train.csv")
    assert os.path.exists("dataprocessed/test.csv")
    assert os.path.exists("dataprocessed/scaler.pkl")

def test_train_test_split_ratio():
    """Test if train-test split is correct."""
    train_df = pd.read_csv("dataprocessed/train.csv")
    test_df = pd.read_csv("dataprocessed/test.csv")
    
    total = len(train_df) + len(test_df)
    assert len(test_df) / total == pytest.approx(0.2, rel=0.01)
