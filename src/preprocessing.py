"""
preprocessing.py
----------------
Handles data loading, text preprocessing, and label encoding
for the SMS Spam Classifier project.

Author: Himanshu Singh
Project: SMS Spam Classifier
"""

import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import DATASET_FILE, LABEL_MAPPING


# ==========================================================
# Download Required NLTK Resources (Only if Missing)
# ==========================================================

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOP_WORDS = set(stopwords.words("english"))


# ==========================================================
# Initialize Stemmer
# ==========================================================

STEMMER = PorterStemmer()


# ==========================================================
# Text Cleaning Function
# ==========================================================

def clean_text(text: str) -> str:
    """
    Cleans a single SMS message.

    Processing Steps
    ----------------
    1. Convert text to lowercase
    2. Remove punctuation, numbers and special characters
    3. Tokenize into words
    4. Remove English stopwords
    5. Apply Porter Stemming

    Parameters
    ----------
    text : str
        Original SMS message.

    Returns
    -------
    str
        Cleaned message.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep only alphabets and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    cleaned_words = [
        STEMMER.stem(word)
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(cleaned_words)


# ==========================================================
# Load Dataset
# ==========================================================

def load_dataset() -> pd.DataFrame:
    """
    Loads the SMS Spam Collection dataset.

    Returns
    -------
    pd.DataFrame
        Dataset containing two columns:
            label
            text

    Raises
    ------
    FileNotFoundError
        If dataset file does not exist.
    """

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    df = pd.read_csv(
        DATASET_FILE,
        encoding="latin-1"
    )

    # Keep only first two useful columns
    df = df.iloc[:, :2]

    df.columns = [
        "label",
        "text"
    ]

    # Remove missing messages
    df.dropna(subset=["text"], inplace=True)

    # Convert to string
    df["text"] = df["text"].astype(str)

    # Remove empty messages
    df = df[df["text"].str.strip() != ""]

    # Reset row numbers
    df.reset_index(drop=True, inplace=True)

    return df


# ==========================================================
# Encode Labels
# ==========================================================

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts labels into numeric values.

    Mapping
    -------
    ham  -> 0
    spam -> 1

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    df["label"] = df["label"].map(LABEL_MAPPING)

    if df["label"].isnull().any():
        raise ValueError(
            "Dataset contains unexpected labels. "
            "Expected only 'ham' and 'spam'."
        )

    return df


# ==========================================================
# Complete Preprocessing Pipeline
# ==========================================================

def preprocess_dataset() -> pd.DataFrame:
    """
    Executes the complete preprocessing pipeline.

    Pipeline
    --------
    Load Dataset
            ↓
    Encode Labels
            ↓
    Clean SMS Text

    Returns
    -------
    pd.DataFrame

    Columns
    -------
    label
    text
    clean_text
    """

    df = load_dataset()

    df = encode_labels(df)

    df["clean_text"] = df["text"].apply(clean_text)

    return df


# ==========================================================
# Module Testing
# ==========================================================

if __name__ == "__main__":

    df = preprocess_dataset()

    print("=" * 60)
    print("SMS Spam Dataset Loaded Successfully")
    print("=" * 60)

    print(f"Dataset Shape : {df.shape}")

    print("\nClass Distribution")
    print(df["label"].value_counts())

    print("\nClass Distribution (%)")
    print(
        (df["label"].value_counts(normalize=True) * 100).round(2)
    )

    print("\nFirst Five Records")
    print(df.head())

    print("\nSample Text Cleaning\n")

    for i in range(5):
        print(f"[{i + 1}] Original : {df['text'].iloc[i]}")
        print(f"    Cleaned : {df['clean_text'].iloc[i]}")
        print("-" * 70)