import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the data into features and target, and then performs a train-test split.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the column to be used as the target (y).

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test
