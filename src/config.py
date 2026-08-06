"""
config.py
---------
Central configuration file for the SpamShield AI project.

Author: Himanshu Singh
Project: SpamShield AI: Intelligent SMS Spam Detection System
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
# Dataset Configuration
# ==========================================================

# Supported dataset file formats
SUPPORTED_DATASET_FORMATS = [
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
]

# Possible column names for labels
LABEL_COLUMNS = [
    "label",
    "target",
    "class",
    "category",
]

# Possible column names for message text
TEXT_COLUMNS = [
    "text",
    "message",
    "sms",
    "content",
]

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

CROSS_VALIDATION_REPORT_FILE = ( OUTPUT_DIR / "cross_validation_report.txt" )

ROC_CURVE_FILE = OUTPUT_DIR / "roc_curve.png"

PRECISION_RECALL_CURVE_FILE = (  OUTPUT_DIR / "precision_recall_curve.png" )

# ==========================================================
# Machine Learning Settings
# ==========================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_FEATURES = 8000

# ==========================================================
# TF-IDF Configuration
# ==========================================================

NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95
SUBLINEAR_TF = True

# ==========================================================
# Model Parameters
# ==========================================================

LOGISTIC_REGRESSION_MAX_ITER = 1000

NAIVE_BAYES_ALPHA = 1.0

# ==========================================================
# Linear SVM Hyperparameter Tuning
# ==========================================================

SVM_PARAM_GRID = {
    "C": [0.1, 0.5, 1, 2, 5]
}

GRID_SEARCH_CV = 5

GRID_SEARCH_SCORING = "f1"

GRID_SEARCH_JOBS = -1

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

PROJECT_NAME = "SpamShield AI: Intelligent SMS Spam Detection System"

PROJECT_VERSION = "2.0.0"

AUTHOR = "Himanshu Singh"