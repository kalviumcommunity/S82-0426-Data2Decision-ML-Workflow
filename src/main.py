import os
import sys

# Ensure the 'src' directory is in the path if running from the root
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_loader import load_data
from preprocessing import preprocess_data
from model import train_model
from evaluate import evaluate_model

def run_pipeline():
    """
    Orchestrates the full Machine Learning pipeline:
    1. Loads data from the data directory.
    2. Preprocesses the data (splitting into train/test).
    3. Trains a Logistic Regression model.
    4. Evaluates the model and prints the accuracy.
    """
    # Define paths (relative to the project root/main.py execution)
    # Since we run from root as 'python src/main.py', the relative path to data is 'data/data.csv'
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
    target_column = 'passed'

    print("--- Starting ML Pipeline ---")

    # 1. Load Data
    print(f"Loading data from: {data_path}")
    df = load_data(data_path)

    # 2. Preprocess Data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df, target_column)

    # 3. Train Model
    print("Training model...")
    trained_model = train_model(X_train, y_train)

    # 4. Evaluate Model
    print("Evaluating model...")
    accuracy = evaluate_model(trained_model, X_test, y_test)

    print(f"\nFinal Model Accuracy: {accuracy:.2f}")
    print("--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    run_pipeline()
