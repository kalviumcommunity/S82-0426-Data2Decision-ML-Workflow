# Modular Machine Learning Pipeline: Professional Workflow

This project implements a professional Machine Learning workflow for predicting student performance. It features a modular design with clear separation between data ingestion, preprocessing, training, evaluation, and isolated inference.

## 📁 Project Structure

```text
project_root/
│
├── data/
│   └── data.csv          # Raw dataset (study_hours, attendance, passed)
│
├── models/
│   └── model.pkl         # Trained binary model (generated after training)
│
├── src/
│   ├── __init__.py       # Package initializer
│   ├── config.py         # Centralized configuration (paths, hyperparameters)
│   ├── data_loader.py    # Robust data loading
│   ├── data_preprocessing.py # Preprocessing and splitting logic
│   ├── feature_engineering.py # Placeholder for feature transformations
│   ├── train.py          # Model training and serialization (fit + save)
│   ├── evaluate.py        # Model performance assessment
│   ├── predict.py        # Isolated inference (load + predict)
│   └── main.py           # Pipeline orchestration
│
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## 🚀 Getting Started

### 1. Installation
Ensure you have the required libraries installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
The project is designed to be run as a module from the root directory:
```bash
python -m src.main
```

## ⚙️ Key Concepts

### Separation of Concerns
- **Training (`train.py`)**: Responsible for learning patterns and persisting the model.
- **Inference (`predict.py`)**: Responsible for loading the persisted model and making predictions on single data points. It **never** retrains the model to avoid data leakage and performance degradation.

### Centralized Configuration
All paths and constants are managed in `src/config.py`, making the project portable across different environments.

## 📊 Model Performance
The pipeline trains a Logistic Regression model on student metrics. Typical accuracy on the sample dataset is high due to the strong relationship between study hours, attendance, and passing results.
