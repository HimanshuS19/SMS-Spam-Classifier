"""
model_training.py
-----------------
Trains machine learning models for SMS Spam Classification.

Author: Himanshu Singh
Project: SMS Spam Classifier
"""

import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

from src.config import (
    LOGISTIC_REGRESSION_MAX_ITER,
    MODEL_DIR,
    MODEL_FILE,
    NAIVE_BAYES_ALPHA,
    RANDOM_STATE,
)

from src.utils import (
    save_pickle,
    load_pickle,
)


# ==========================================================
# Train Naive Bayes
# ==========================================================

def train_naive_bayes(X_train, y_train):
    """
    Train a Multinomial Naive Bayes classifier.

    Returns
    -------
    tuple
        (model, training_time)
    """

    print("\nTraining Multinomial Naive Bayes...")

    start_time = time.perf_counter()

    model = MultinomialNB(
        alpha=NAIVE_BAYES_ALPHA,
    )

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print(f"✔ Completed in {training_time:.4f} seconds")

    return model, training_time


# ==========================================================
# Train Logistic Regression
# ==========================================================

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression classifier.

    Returns
    -------
    tuple
        (model, training_time)
    """

    print("\nTraining Logistic Regression...")

    start_time = time.perf_counter()

    model = LogisticRegression(
        max_iter=LOGISTIC_REGRESSION_MAX_ITER,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print(f"✔ Completed in {training_time:.4f} seconds")

    return model, training_time


# ==========================================================
# Compare Models
# ==========================================================

def compare_models(
    nb_model,
    lr_model,
    X_test,
    y_test,
):
    """
    Compare models using accuracy.

    Returns
    -------
    tuple
        (
            best_model,
            best_model_name,
            scores
        )
    """

    nb_accuracy = accuracy_score(
        y_test,
        nb_model.predict(X_test),
    )

    lr_accuracy = accuracy_score(
        y_test,
        lr_model.predict(X_test),
    )

    scores = {
        "Naive Bayes": nb_accuracy,
        "Logistic Regression": lr_accuracy,
    }

    print("\nModel Comparison")
    print("-" * 40)

    for name, score in scores.items():
        print(f"{name:<25} : {score:.4f}")

    if lr_accuracy >= nb_accuracy:
        print("\n✔ Best Model: Logistic Regression")
        return lr_model, "Logistic Regression", scores

    print("\n✔ Best Model: Naive Bayes")
    return nb_model, "Naive Bayes", scores


# ==========================================================
# Save Model
# ==========================================================

def save_model(model):
    """
    Save trained model.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        save_pickle(
            model,
            MODEL_FILE,
        )

        print("\nSaving Model...")
        print("✔ Model Saved Successfully")
        print(f"Location : {MODEL_FILE}")

    except Exception as e:
        raise RuntimeError(
            f"Unable to save model.\n{e}"
        ) from e


# ==========================================================
# Load Model
# ==========================================================

def load_model():
    """
    Load trained model.

    Returns
    -------
    object
        Trained machine learning model.
    """

    try:
        return load_pickle(MODEL_FILE)

    except Exception as e:
        raise RuntimeError(
            f"Unable to load model.\n{e}"
        ) from e


# ==========================================================
# Module Testing
# ==========================================================

if __name__ == "__main__":

    from src.preprocessing import preprocess_dataset
    from src.feature_engineering import (
        create_train_test_split,
        vectorize_data,
    )

    print("=" * 60)
    print("SMS Spam Classifier - Model Training")
    print("=" * 60)

    # Load Dataset
    df = preprocess_dataset()

    # Train/Test Split
    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    ) = create_train_test_split(df)

    # Feature Engineering
    (
        X_train,
        X_test,
        vectorizer,
    ) = vectorize_data(
        X_train_text,
        X_test_text,
    )

    # Train Models
    nb_model, nb_time = train_naive_bayes(
        X_train,
        y_train,
    )

    lr_model, lr_time = train_logistic_regression(
        X_train,
        y_train,
    )

    # Compare Models
    (
        best_model,
        best_name,
        scores,
    ) = compare_models(
        nb_model,
        lr_model,
        X_test,
        y_test,
    )

    # Save Best Model
    save_model(best_model)

    print("\n" + "=" * 60)
    print("Training Completed Successfully")
    print("=" * 60)

    print(f"Selected Model      : {best_name}")
    print(f"Naive Bayes Time    : {nb_time:.4f} sec")
    print(f"Logistic Regression : {lr_time:.4f} sec")