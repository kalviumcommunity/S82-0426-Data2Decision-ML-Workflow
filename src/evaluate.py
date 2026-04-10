from sklearn.metrics import accuracy_score, classification_report
from src.config import SKILL_LABELS

def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluates multi-label classification performance.
    """
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-skill reporting
    report = classification_report(y_test, y_pred, target_names=SKILL_LABELS, zero_division=0)
    
    return accuracy, report
