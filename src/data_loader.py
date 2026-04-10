import pandas as pd
from src.config import TARGET_COLUMN, ALL_FEATURES, SKILL_LABELS

def load_data(path: str) -> pd.DataFrame:
    """
    Loads raw dataset and validates core columns.
    """
    try:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Dataset at {path} is empty.")
            
        # Assignment: Validation that target exists and is not in features
        # We check SKILL_LABELS for the actual multi-label structure
        target_cols = SKILL_LABELS
        assert all(c in df.columns for c in target_cols), f"Target labels {target_cols} missing in CSV"
            
        assert all(c in df.columns for c in ALL_FEATURES), f"Feature columns {ALL_FEATURES} missing in CSV"
        
        print("\n📊 DATA LOADING & VALIDATION")
        print(f"Features: {df[ALL_FEATURES].shape}")
        
        # Target Distribution (Multi-label summary)
        print("Target distribution (Label Counts):")
        print(df[target_cols].sum())
            
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {path}")
    except Exception as e:
        raise Exception(f"Validation Error: {e}")
