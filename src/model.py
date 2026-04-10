import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Trains a Logistic Regression model on the provided training data.

    Parameters:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target.

    Returns:
    LogisticRegression: The trained model object.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
