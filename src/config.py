"""
config.py
----------
Central configuration file for the SMS Spam Classifier project.
Contains project paths and model settings.
"""

from pathlib import Path

# ==========================================================
# Project Directories
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

# ==========================================================
# Dataset
# ==========================================================

DATASET_FILE = DATASET_DIR / "spam.csv"

# ==========================================================
# Saved Models
# ==========================================================

MODEL_FILE = MODEL_DIR / "spam_model.pkl"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"

# ==========================================================
# Output Images
# ==========================================================

CLASS_DISTRIBUTION = OUTPUT_DIR / "class_distribution.png"
SPAM_WORDCLOUD = OUTPUT_DIR / "spam_wordcloud.png"
HAM_WORDCLOUD = OUTPUT_DIR / "ham_wordcloud.png"
CONFUSION_MATRIX = OUTPUT_DIR / "confusion_matrix.png"
MODEL_COMPARISON = OUTPUT_DIR / "model_comparison.png"

# ==========================================================
# Machine Learning Settings
# ==========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_FEATURES = 5000

# ==========================================================
# Label Mapping
# ==========================================================

LABEL_MAPPING = {
    "ham": 0,
    "spam": 1
}

LABEL_NAMES = ["Ham", "Spam"]