"""
preprocessing.py
----------------
Handles dataset loading, preprocessing, text cleaning,
dataset merging and label encoding.

Author: Himanshu Singh
Project: SMS Spam Classifier
"""

import re
import warnings

import nltk
import pandas as pd

from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import (
    DATASET_DIR,
    LABEL_MAPPING,
)


# ==========================================================
# Download NLTK Resources (Only if Missing)
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
# Supported Dataset Formats
# ==========================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
}


# ==========================================================
# Text Cleaning
# ==========================================================

def clean_text(text: str) -> str:
    """
    Clean a single SMS message.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    cleaned_words = [
        STEMMER.stem(word)
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(cleaned_words)


# ==========================================================
# Scan Dataset Folder
# ==========================================================

def scan_dataset_directory():
    """
    Returns every supported dataset inside dataset folder.
    """

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_DIR}"
        )

    dataset_files = sorted(
        [
            file
            for file in DATASET_DIR.iterdir()
            if file.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
    )

    if not dataset_files:
        raise FileNotFoundError(
            "No supported datasets found."
        )

    return dataset_files


# ==========================================================
# Read Single Dataset
# ==========================================================

def load_single_dataset(file_path: Path):
    """
    Load one dataset.
    """

    suffix = file_path.suffix.lower()

    try:

        if suffix == ".csv":

            try:
                df = pd.read_csv(file_path)

            except UnicodeDecodeError:

                try:
                    df = pd.read_csv(
                        file_path,
                        encoding="latin-1",
                    )

                except UnicodeDecodeError:

                    df = pd.read_csv(
                        file_path,
                        encoding="cp1252",
                    )

        elif suffix in [".xlsx", ".xls"]:

            df = pd.read_excel(file_path)

        elif suffix == ".json":

            df = pd.read_json(file_path)

        else:

            raise ValueError(
                f"Unsupported file type: {suffix}"
            )

        print(f"✓ {file_path.name:<25} Rows : {len(df)}")

        return df

    except Exception as e:

        warnings.warn(
            f"Skipped {file_path.name}\n{e}"
        )

        return None


# ==========================================================
# Standardize Column Names
# ==========================================================

def standardize_columns(df: pd.DataFrame):
    """
    Convert different dataset formats into:

    label
    text
    """

    columns = {
        col.lower().strip(): col
        for col in df.columns
    }

    label_candidates = [
        "label",
        "target",
        "class",
        "category",
    ]

    text_candidates = [
        "text",
        "message",
        "sms",
        "content",
    ]

    label_column = None
    text_column = None

    for col in label_candidates:
        if col in columns:
            label_column = columns[col]
            break

    for col in text_candidates:
        if col in columns:
            text_column = columns[col]
            break

    if label_column is None or text_column is None:
        raise ValueError(
            "Could not detect label/text columns."
        )

    df = df[[label_column, text_column]].copy()

    df.columns = [
        "label",
        "text",
    ]

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    return df

# ==========================================================
# Load & Merge All Datasets
# ==========================================================

def load_dataset() -> pd.DataFrame:
    """
    Load and merge every supported dataset inside
    the dataset directory.
    """

    print("=" * 60)
    print("Scanning Dataset Folder")
    print("=" * 60)

    dataset_files = scan_dataset_directory()

    datasets = []

    total_rows = 0

    for file in dataset_files:

        df = load_single_dataset(file)

        if df is None:
            continue

        try:
            df = standardize_columns(df)

        except Exception as e:

            warnings.warn(
                f"Skipped {file.name}\n{e}"
            )
            continue

        datasets.append(df)

        total_rows += len(df)

    if not datasets:
        raise ValueError(
            "No valid datasets available."
        )

    df = pd.concat(
        datasets,
        ignore_index=True,
    )

    before = len(df)

    # Remove empty messages
    df.dropna(
        subset=["text"],
        inplace=True,
    )

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"] != ""
    ]

    # Remove duplicate SMS messages
    df.drop_duplicates(
        subset="text",
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    duplicates_removed = before - len(df)

    print("\n" + "-" * 60)
    print(f"Datasets Found      : {len(datasets)}")
    print(f"Original Rows       : {total_rows}")
    print(f"Duplicates Removed  : {duplicates_removed}")
    print(f"Final Dataset Rows  : {len(df)}")
    print("-" * 60)

    return df


# ==========================================================
# Encode Labels
# ==========================================================

def encode_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert:

    ham  -> 0
    spam -> 1
    """

    df = df.copy()

    df["label"] = (
        df["label"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    invalid_labels = set(
        df["label"].unique()
    ) - set(LABEL_MAPPING.keys())

    if invalid_labels:

        raise ValueError(
            "Unexpected labels found:\n"
            f"{sorted(invalid_labels)}"
        )

    df["label"] = df["label"].map(
        LABEL_MAPPING
    )

    return df


# ==========================================================
# Complete Preprocessing Pipeline
# ==========================================================

def preprocess_dataset():
    """
    Complete preprocessing pipeline.

    Returns
    -------
    DataFrame

    Columns
    -------
    label
    text
    clean_text
    """

    df = load_dataset()

    print("\nEncoding Labels...")

    df = encode_labels(df)

    print("Cleaning SMS Messages...")

    df["clean_text"] = (
        df["text"]
        .apply(clean_text)
    )

    print("✔ Preprocessing Completed")

    return df


# ==========================================================
# Module Testing
# ==========================================================

if __name__ == "__main__":

    df = preprocess_dataset()

    print("\n" + "=" * 60)
    print("SMS Spam Dataset Ready")
    print("=" * 60)

    print(f"Dataset Shape : {df.shape}")

    print("\nClass Distribution")

    print(
        df["label"]
        .value_counts()
    )

    print("\nClass Distribution (%)")

    print(
        (
            df["label"]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    print("\nFirst Five Records")

    print(
        df.head()
    )

    print("\nSample Cleaning\n")

    for i in range(
        min(5, len(df))
    ):

        print(
            f"[{i+1}] Original : "
            f"{df['text'].iloc[i]}"
        )

        print(
            f"    Cleaned : "
            f"{df['clean_text'].iloc[i]}"
        )

        print("-" * 70)