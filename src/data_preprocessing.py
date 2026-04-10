import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from src.config import RANDOM_STATE, TEST_SIZE

def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans data and splits it into training and testing sets.
    
    Parameters:
    df (pd.DataFrame): Raw data.
    target_column (str): Name of target column.
    
    Returns:
    Tuple: X_train, X_test, y_train, y_test
    """
    # Simple feature/target split
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test