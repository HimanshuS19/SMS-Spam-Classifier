"""
feature_engineering.py
----------------------
Handles train-test splitting and TF-IDF feature engineering
for the SpamShield AI: Intelligent SMS Spam Detection System project.

Author: Himanshu Singh
Project: SpamShield AI: Intelligent SMS Spam Detection System
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.config import (
    MAX_FEATURES,
    NGRAM_RANGE,
    MIN_DF,
    MAX_DF,
    SUBLINEAR_TF,
    MODEL_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    VECTORIZER_FILE,
)


# ==========================================================
# Train-Test Split
# ==========================================================

def create_train_test_split(
    df: pd.DataFrame
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.

    Returns
    -------
    tuple
        X_train_text
        X_test_text
        y_train
        y_test
    """

    if df.empty:
        raise ValueError("Dataset is empty.")

    required_columns = {"clean_text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}"
        )

    print("\nCreating Train/Test Split...")

    X = df["clean_text"]
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("✔ Train/Test Split Completed")

    return X_train_text, X_test_text, y_train, y_test


# ==========================================================
# TF-IDF Vectorizer
# ==========================================================

def build_tfidf_vectorizer() -> TfidfVectorizer:
    """
    Creates an optimized TF-IDF Vectorizer.
    """

    print("\nBuilding TF-IDF Vectorizer...")

    print("-" * 40)
    print("TF-IDF Configuration")
    print("-" * 40)
    print(f"Max Features : {MAX_FEATURES}")
    print(f"N-Grams      : {NGRAM_RANGE}")
    print(f"Min DF       : {MIN_DF}")
    print(f"Max DF       : {MAX_DF}")
    print(f"Sublinear TF : {SUBLINEAR_TF}")

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=SUBLINEAR_TF,
    )

    print("\n✔ Vectorizer Created")

    return vectorizer

# ==========================================================
# Vectorize Data
# ==========================================================

def vectorize_data(
    X_train_text: pd.Series,
    X_test_text: pd.Series,
) -> tuple:
    """
    Fits TF-IDF on training data and transforms
    both training and testing datasets.
    """

    vectorizer = build_tfidf_vectorizer()

    print("\nVectorizing Text...")

    X_train = vectorizer.fit_transform(X_train_text)

    X_test = vectorizer.transform(X_test_text)

    print("✔ Text Vectorization Completed")

    print("\nVocabulary Statistics")
    print("-" * 40)
    print(f"Vocabulary Size : {len(vectorizer.vocabulary_)}")
    print(f"Training Shape  : {X_train.shape}")
    print(f"Testing Shape   : {X_test.shape}")

    return (
        X_train,
        X_test,
        vectorizer,
    )

# ==========================================================
# Save Vectorizer
# ==========================================================

def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    """
    Saves the trained TF-IDF vectorizer.

    Parameters
    ----------
    vectorizer : TfidfVectorizer
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        joblib.dump(
            vectorizer,
            VECTORIZER_FILE,
        )

        print("\nSaving Vectorizer...")
        print("✔ Vectorizer Saved Successfully")
        print(f"Location : {VECTORIZER_FILE}")

    except Exception as e:
        raise RuntimeError(
            f"Unable to save vectorizer.\n{e}"
        )


# ==========================================================
# Load Vectorizer
# ==========================================================

def load_vectorizer() -> TfidfVectorizer:
    """
    Loads a previously saved TF-IDF vectorizer.

    Returns
    -------
    TfidfVectorizer
    """

    if not Path(VECTORIZER_FILE).exists():
        raise FileNotFoundError(
            f"Vectorizer not found:\n{VECTORIZER_FILE}"
        )

    try:

        vectorizer = joblib.load(
            VECTORIZER_FILE
        )

        return vectorizer

    except Exception as e:
        raise RuntimeError(
            f"Unable to load vectorizer.\n{e}"
        )


# ==========================================================
# Module Testing
# ==========================================================

if __name__ == "__main__":

    from src.preprocessing import preprocess_dataset

    df = preprocess_dataset()

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    ) = create_train_test_split(df)

    (
        X_train,
        X_test,
        vectorizer,
    ) = vectorize_data(
        X_train_text,
        X_test_text,
    )

    print("\n" + "=" * 60)
    print("Feature Engineering Completed Successfully")
    print("=" * 60)

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")

    print(f"\nTF-IDF Features : {X_train.shape[1]}")

    print(f"\nTraining Matrix Shape : {X_train.shape}")
    print(f"Testing Matrix Shape  : {X_test.shape}")

    save_vectorizer(vectorizer)