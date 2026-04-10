import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from src.config import MODEL_PATH

def train_model(X_train_tfidf, y_train):
    """
    Trains a Multi-label classifier (OneVsRest with RandomForest).
    """
    base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    model = OneVsRestClassifier(base_clf)
    
    model.fit(X_train_tfidf, y_train)
    
    # Save the model
    joblib.dump(model, MODEL_PATH)
    print(f"Skill Classifier saved to {MODEL_PATH}")
    
    return model