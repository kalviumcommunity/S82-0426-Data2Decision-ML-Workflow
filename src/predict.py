import joblib
import pandas as pd
from typing import Dict, Any
from src.config import MODEL_PATH, FEATURES

def predict(sample_dict: Dict[str, Any]) -> Any:
    """
    Loads the saved model and predicts on a new data point.
    
    Parameters:
    sample_dict (Dict[str, Any]): Input data as a dictionary.
    
    Returns:
    Any: Model prediction.
    """
    # Load the model (Isolation: No retraining here!)
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first.")

    # Convert to DataFrame
    df = pd.DataFrame([sample_dict])

    # Ensure feature order matches training
    df = df[FEATURES]

    # Predict
    prediction = model.predict(df)
    
    return prediction[0]