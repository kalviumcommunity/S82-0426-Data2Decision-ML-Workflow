import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def evaluate_model(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Evaluates a trained model using test data.
    
    Parameters:
    model (LogisticRegression): Trained model object.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target.
    
    Returns:
    float: Accuracy score.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy
