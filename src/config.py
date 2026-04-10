# Configuration for Project Description Analyzer (NLP)

# Data paths
RAW_DATA_PATH = "data/raw/projects.csv"
PROCESSED_DATA_PATH = "data/processed/"
DATA_PATH = RAW_DATA_PATH  # Main path used by data_loader

MODEL_PATH = "models/skill_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

# --- Assignment: Feature Columns and Target Variables ---
# Skill labels (Target columns)
SKILL_LABELS = ["Python", "SQL", "React", "Machine_Learning", "Flask"]

TARGET_COLUMN = SKILL_LABELS # Multi-label target: technologies predicted

NUMERICAL_FEATURES = []  # No numeric features
CATEGORICAL_FEATURES = []  # No categorical features
TEXT_FEATURES = ["project_description"]
EXCLUDED_COLUMNS = [
    "project_id",  # unique identifier - avoids potential overfitting
    "manual_tags"  # potential leakage - contains target information
]

ALL_FEATURES = TEXT_FEATURES

# NLP Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2