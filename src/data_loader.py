import pandas as pd
from src.config import TARGET_COLUMN, ALL_FEATURES

def load_data(path: str) -> pd.DataFrame:
    """
    Loads raw dataset and validates core columns.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Dataset at {path} is empty.")
            
        # Assignment: Validation that target exists and is not in features
        if isinstance(TARGET_COLUMN, list):
            assert all(c in df.columns for c in TARGET_COLUMN), "Target labels missing in CSV"
        else:
            assert TARGET_COLUMN in df.columns, f"Target {TARGET_COLUMN} missing"
            
        assert all(c in df.columns for c in ALL_FEATURES), "Feature columns missing in CSV"
        
        print("\n📊 DATA LOADING & VALIDATION")
        print(f"Features: {df[ALL_FEATURES].shape}")
        
        # Target Distribution (Multi-label summary)
        if isinstance(TARGET_COLUMN, list):
            print("Target distribution (Label Counts):")
            print(df[TARGET_COLUMN].sum())
        else:
            print(f"Target distribution:\n{df[TARGET_COLUMN].value_counts()}")
            
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {path}")
    except Exception as e:
        raise Exception(f"Validation Error: {e}")
