# Configuration for Project Description Analyzer (NLP)

DATA_PATH = "data/projects.csv"
MODEL_PATH = "models/skill_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Skill labels (Target columns)
SKILL_LABELS = ["Python", "SQL", "React", "Machine_Learning", "Flask"]

# NLP Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2