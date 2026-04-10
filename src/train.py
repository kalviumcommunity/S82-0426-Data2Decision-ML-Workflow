import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.config import MODEL_PATH

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """
    Trains a Logistic Regression model and saves it to a file.
    
    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    
    Returns:
    LogisticRegression: Trained model object.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

    return model