import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluates the model on the test data and returns the accuracy score.

    Parameters:
    model (LogisticRegression): The trained model object.
    X_test (pd.DataFrame): The test features.
    y_test (pd.Series): The test target.

    Returns:
    float: The accuracy score as a decimal between 0 and 1.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
