# Configuration for Project Description Analyzer (NLP)

# Data paths
RAW_DATA_PATH = "data/raw/projects.csv"
PROCESSED_DATA_PATH = "data/processed/"
DATA_PATH = RAW_DATA_PATH  # Main path used by data_loader

MODEL_PATH = "models/skill_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# Skill labels (Target columns)
SKILL_LABELS = ["Python", "SQL", "React", "Machine_Learning", "Flask"]

# NLP Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2