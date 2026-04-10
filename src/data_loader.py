import pandas as pd
from typing import Protocol

def load_data(path: str) -> pd.DataFrame:
    """
    Loads raw dataset from a CSV file.
    
    Parameters:
    path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Dataset at {path} is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {path}")
    except Exception as e:
        raise Exception(f"Unexpected error loading data: {e}")
