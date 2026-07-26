"""
config.py
---------
Central configuration file for the SMS Spam Classifier project.

Author: Himanshu Singh
Project: SMS Spam Classifier
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
# Output Files
# ==========================================================

CLASS_DISTRIBUTION_FILE = OUTPUT_DIR / "class_distribution.png"

SPAM_WORDCLOUD_FILE = OUTPUT_DIR / "spam_wordcloud.png"
HAM_WORDCLOUD_FILE = OUTPUT_DIR / "ham_wordcloud.png"

CONFUSION_MATRIX_FILE = OUTPUT_DIR / "confusion_matrix.png"

MODEL_COMPARISON_FILE = OUTPUT_DIR / "model_comparison.png"

EVALUATION_REPORT_FILE = OUTPUT_DIR / "evaluation_report.txt"

# ==========================================================
# Machine Learning Settings
# ==========================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

MAX_FEATURES = 5000

# ==========================================================
# Model Parameters
# ==========================================================

LOGISTIC_REGRESSION_MAX_ITER = 1000

NAIVE_BAYES_ALPHA = 1.0

# ==========================================================
# Label Mapping
# ==========================================================

LABEL_MAPPING = {
    "ham": 0,
    "spam": 1,
}

LABEL_NAMES = [
    "Ham",
    "Spam",
]

# ==========================================================
# Flask Settings
# ==========================================================

HOST = "127.0.0.1"

PORT = 5000

DEBUG = True

# ==========================================================
# Application Information
# ==========================================================

PROJECT_NAME = "SMS Spam Classifier"

PROJECT_VERSION = "1.0.0"

AUTHOR = "Himanshu Singh"