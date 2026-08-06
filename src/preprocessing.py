"""
preprocessing.py
----------------
Handles dataset loading, preprocessing, text cleaning,
dataset merging and label encoding.

Author: Himanshu Singh
Project: SpamShield AI: Intelligent SMS Spam Detection System
"""

import re
import warnings

from pathlib import Path

import nltk
import pandas as pd

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
    Returns all supported datasets inside dataset folder.
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

    Supports:
    - CSV
    - TSV (.csv with tab separator)
    - Excel
    - JSON
    """

    suffix = file_path.suffix.lower()

    try:

        if suffix == ".csv":

            encodings = [
                "utf-8",
                "latin-1",
                "cp1252",
            ]

            df = None

            for encoding in encodings:

                # --------------------------------------------------
                # Try TAB-separated first
                # --------------------------------------------------
                try:

                    temp = pd.read_csv(
                        file_path,
                        sep="\t",
                        header=None,
                        names=["label", "text"],
                        encoding=encoding,
                        engine="python",
                        quoting=3,
                        keep_default_na=False,
                        on_bad_lines="skip",
                    )

                    valid = (
                        temp["label"]
                        .astype(str)
                        .str.lower()
                        .str.strip()
                        .isin(["ham", "spam"])
                        .mean()
                    )

                    if valid >= 0.90:
                        df = temp
                        break

                except Exception:
                    pass

                # --------------------------------------------------
                # Try normal CSV
                # --------------------------------------------------
                try:

                    temp = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        engine="python",
                        on_bad_lines="skip",
                    )

                    df = temp
                    break

                except Exception:
                    pass

            if df is None:
                raise ValueError("Unable to read CSV file.")

        elif suffix in [".xlsx", ".xls"]:

            df = pd.read_excel(file_path)

        elif suffix == ".json":

            df = pd.read_json(file_path)

        else:

            raise ValueError(
                f"Unsupported file type: {suffix}"
            )

        print(
            f"✓ {file_path.name:<25} Rows : {len(df)}"
        )

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
    Convert every dataset into:

        label
        text
    """

    df = df.copy()

    # --------------------------------------------------
    # Already standardised
    # --------------------------------------------------

    if {"label", "text"}.issubset(df.columns):

        return df[["label", "text"]]

    # --------------------------------------------------
    # Normalise column names
    # --------------------------------------------------

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    # --------------------------------------------------
    # Standard column names
    # --------------------------------------------------

    label_candidates = [
        "label",
        "target",
        "class",
        "category",
        "type",
        "v1",
    ]

    text_candidates = [
        "text",
        "message",
        "sms",
        "content",
        "msg",
        "v2",
    ]

    label_column = None
    text_column = None

    for col in label_candidates:
        if col in df.columns:
            label_column = col
            break

    for col in text_candidates:
        if col in df.columns:
            text_column = col
            break

    if label_column and text_column:

        new_df = df[
            [label_column, text_column]
        ].copy()

        new_df.columns = [
            "label",
            "text",
        ]

        return new_df

    # --------------------------------------------------
    # Detect if first column header itself contains
    # "ham\tmessage..."
    # --------------------------------------------------

    first_header = str(df.columns[0])

    if "\t" in first_header:

        rows = []

        first_line = "\t".join(df.columns.astype(str))

        lines = [first_line]

        for row in df.itertuples(index=False):
            lines.append("\t".join(map(str, row)))

        for line in lines:

            parts = line.split("\t", 1)

            if len(parts) != 2:
                continue

            label = parts[0].strip().lower()

            if label not in ["ham", "spam"]:
                continue

            rows.append([
                label,
                parts[1].strip(),
            ])

        if rows:

            return pd.DataFrame(
                rows,
                columns=["label", "text"],
            )

    # --------------------------------------------------
    # One-column dataset
    # --------------------------------------------------

    if len(df.columns) == 1:

        extracted = (
            df.iloc[:, 0]
            .astype(str)
            .str.extract(
                r"^(ham|spam)\s+(.*)$",
                flags=re.IGNORECASE,
            )
        )

        if extracted.notna().all().all():

            extracted.columns = [
                "label",
                "text",
            ]

            return extracted

    # --------------------------------------------------
    # First column is label
    # --------------------------------------------------

    first = (
        df.iloc[:, 0]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if first.isin(["ham", "spam"]).mean() >= 0.80:

        message = (
            df.iloc[:, 1:]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )

        return pd.DataFrame(
            {
                "label": first,
                "text": message,
            }
        )

    raise ValueError(
        "Could not detect dataset format."
    )
# ==========================================================
# Load & Merge All Datasets
# ==========================================================

def load_dataset() -> pd.DataFrame:
    """
    Load every supported dataset from dataset folder,
    automatically merge them and remove duplicates.
    """

    print("=" * 60)
    print("Scanning Dataset Folder")
    print("=" * 60)

    dataset_files = scan_dataset_directory()

    datasets = []

    total_rows = 0

    for file in dataset_files:

        df = load_single_dataset(file)

        # --------------------------------------------------
        # Fallback Parser
        #
        # Supports datasets like:
        #
        # ham Hello...
        # spam Win money...
        # hamHello...
        # spamFree...
        # --------------------------------------------------

        if (
            file.suffix.lower() == ".csv"
            and (
                df is None
                or len(df.columns) == 1
            )
        ):

            try:

                rows = []

                lines = file.read_text(
                    encoding="latin-1",
                    errors="ignore",
                ).splitlines()

                for line in lines:

                    line = line.strip()

                    if not line:
                        continue

                    match = re.match(
                        r"^(ham|spam)\s*(.+)$",
                        line,
                        flags=re.IGNORECASE,
                    )

                    if match:

                        rows.append([
                            match.group(1).lower(),
                            match.group(2).strip(),
                        ])

                if rows:

                    df = pd.DataFrame(
                        rows,
                        columns=[
                            "label",
                            "text",
                        ],
                    )

                    print(
                        f"✓ {file.name:<25} Rows : {len(df)} (fallback parser)"
                    )

            except Exception:
                pass

        if df is None:
            continue

        try:

            df = standardize_columns(df)

        except Exception as e:

            warnings.warn(
                f"Skipped {file.name}\n{e}"
            )

            continue

        # ------------------------------
        # Clean label column
        # ------------------------------

        df["label"] = (
            df["label"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # ------------------------------
        # Clean text column
        # ------------------------------

        df["text"] = (
            df["text"]
            .astype(str)
            .str.strip()
        )

        # Keep only ham/spam

        df = df[
            df["label"].isin(
                LABEL_MAPPING.keys()
            )
        ]

        # Remove empty messages

        df = df[
            df["text"] != ""
        ]

        datasets.append(df)

        total_rows += len(df)

    # ------------------------------------------------------
    # No dataset found
    # ------------------------------------------------------

    if not datasets:

        raise ValueError(
            "No valid datasets available."
        )

    # ------------------------------------------------------
    # Merge datasets
    # ------------------------------------------------------

    df = pd.concat(
        datasets,
        ignore_index=True,
    )

    before = len(df)

    # Remove duplicate SMS

    df.drop_duplicates(

        subset="text",

        inplace=True,

    )

    df.reset_index(

        drop=True,

        inplace=True,

    )

    duplicates_removed = before - len(df)

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------

    print("\n" + "-" * 60)

    print(
        f"Datasets Found      : {len(datasets)}"
    )

    print(
        f"Original Rows       : {total_rows}"
    )

    print(
        f"Duplicates Removed  : {duplicates_removed}"
    )

    print(
        f"Final Dataset Rows  : {len(df)}"
    )

    print("-" * 60)

    return df


# ==========================================================
# Encode Labels
# ==========================================================

def encode_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode labels into numeric values.

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

    invalid_labels = (
        set(df["label"].unique())
        - set(LABEL_MAPPING.keys())
    )

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

    Steps
    -----
    1. Load all datasets
    2. Merge datasets
    3. Encode labels
    4. Clean text
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

    print("\nSample Records")

    print(
        df.head()
    )

    print("\nSample Cleaning\n")

    for i in range(min(5, len(df))):

        print(
            f"[{i+1}] Original : "
            f"{df['text'].iloc[i]}"
        )

        print(
            f"    Cleaned : "
            f"{df['clean_text'].iloc[i]}"
        )

        print("-" * 70)

    print("\nDataset Summary")

    print("-" * 60)

    print(f"Total Messages : {len(df)}")

    print(
        f"Ham Messages   : {(df['label'] == 0).sum()}"
    )

    print(
        f"Spam Messages  : {(df['label'] == 1).sum()}"
    )

    print(
        f"Spam Ratio     : "
        f"{((df['label'] == 1).mean() * 100):.2f}%"
    )

    print("-" * 60)

    print("\n✔ preprocessing.py executed successfully.")