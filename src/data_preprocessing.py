import pandas as pd
import re
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from src.config import RANDOM_STATE, TEST_SIZE, SKILL_LABELS

def clean_text(text: str) -> str:
    """
    Cleans the project description text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses project data for multi-label classification.
    """
    # Clean descriptions
    df['description'] = df['description'].apply(clean_text)
    
    X = df['description']
    y = df[SKILL_LABELS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test