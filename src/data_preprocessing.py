import pandas as pd
import re
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from src.config import RANDOM_STATE, TEST_SIZE, SKILL_LABELS, ALL_FEATURES, TARGET_COLUMN

def clean_text(text: str) -> str:
    """
    Cleans the project description text.
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses project data for multi-label classification.
    """
    # 1. Validation Logic (Assignment Requirement)
    # Check if target columns exist
    if isinstance(TARGET_COLUMN, list):
        target_check = all(col in df.columns for col in TARGET_COLUMN)
        target_name = "Skill Labels"
    else:
        target_check = TARGET_COLUMN in df.columns
        target_name = TARGET_COLUMN

    assert target_check, f"Target column(s) {TARGET_COLUMN} not found in dataset."
    
    # Ensure features and target are separate
    for feat in ALL_FEATURES:
        assert feat not in (TARGET_COLUMN if isinstance(TARGET_COLUMN, list) else [TARGET_COLUMN]), \
            f"Feature column '{feat}' cannot be part of the Target Variable."

    # 2. X and y Separation
    X = df[ALL_FEATURES[0]] # For NLP, we typically use the main text feature
    y = df[TARGET_COLUMN]

    print(f"✅ Features Shape: {df[ALL_FEATURES].shape}")
    print(f"🎯 Target Shape: {y.shape}")
    
    # Clean descriptions (using the correct feature name)
    X = X.apply(clean_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test