"""
utils.py
--------
Common utility functions for the SpamShield AI: Intelligent SMS Spam Detection System.

These helper functions are shared across multiple modules
to avoid code duplication.

Author: Himanshu Singh
Project: SpamShield AI: Intelligent SMS Spam Detection System
"""

from pathlib import Path
from datetime import datetime
import joblib


# ==========================================================
# Pickle Utilities
# ==========================================================

def save_pickle(obj, file_path):
    """
    Save any Python object using joblib.

    Parameters
    ----------
    obj : object
        Python object to save.

    file_path : str | Path
        Destination path.
    """

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        joblib.dump(obj, file_path)

    except Exception as e:
        raise RuntimeError(
            f"Unable to save object.\n{e}"
        )


def load_pickle(file_path):
    """
    Load a saved joblib object.

    Parameters
    ----------
    file_path : str | Path

    Returns
    -------
    object
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    try:
        return joblib.load(file_path)

    except Exception as e:
        raise RuntimeError(
            f"Unable to load object.\n{e}"
        )


# ==========================================================
# Input Validation
# ==========================================================

def validate_input(message):
    """
    Validate user input.

    Parameters
    ----------
    message : str

    Returns
    -------
    str
        Cleaned message.
    """

    if not isinstance(message, str):
        raise TypeError(
            "Input must be a string."
        )

    message = message.strip()

    if len(message) == 0:
        raise ValueError(
            "Message cannot be empty."
        )

    return message


# ==========================================================
# Formatting Utilities
# ==========================================================

def format_percentage(value):
    """
    Format probability values.

    Example
    -------
    98.45678 -> 98.46
    """

    return round(float(value), 2)


# ==========================================================
# Timestamp
# ==========================================================

def get_timestamp():
    """
    Return current timestamp.

    Returns
    -------
    str
    """

    return datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )


# ==========================================================
# Directory Utility
# ==========================================================

def create_directory(path):
    """
    Create directory if it doesn't exist.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


# ==========================================================
# Console Printing
# ==========================================================

def print_header(title, width=60):
    """
    Print formatted section header.
    """

    print("=" * width)
    print(title)
    print("=" * width)


def print_success(message):
    """
    Print success message.
    """

    print(f"✔ {message}")


def print_error(message):
    """
    Print error message.
    """

    print(f"✖ {message}")


def print_info(message):
    """
    Print informational message.
    """

    print(f"➜ {message}")


# ==========================================================
# Module Test
# ==========================================================

if __name__ == "__main__":

    print_header("Testing Utility Functions")

    print_success("Utility module loaded")

    print()

    print("Timestamp")
    print(get_timestamp())

    print()

    print("Percentage")
    print(format_percentage(98.7654321))

    print()

    msg = validate_input("   Hello World   ")

    print("Validated Input")
    print(msg)

    print()

    print_success("All utility functions working correctly.")