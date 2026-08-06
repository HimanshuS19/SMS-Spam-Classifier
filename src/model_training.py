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
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from src.config import (
    LOGISTIC_REGRESSION_MAX_ITER,
    MODEL_DIR,
    MODEL_FILE,
    NAIVE_BAYES_ALPHA,
    RANDOM_STATE,
    SVM_PARAM_GRID,
    GRID_SEARCH_CV,
    GRID_SEARCH_SCORING,
    GRID_SEARCH_JOBS,
    OUTPUT_DIR,
    CROSS_VALIDATION_REPORT_FILE,
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
# Train Optimized Linear SVM
# ==========================================================

def train_linear_svm(X_train, y_train):
    """
    Train an optimized Linear Support Vector Machine
    using GridSearchCV.

    Returns
    -------
    tuple
        (best_model, training_time)
    """

    print("\nTraining Optimized Linear SVM...")

    start_time = time.perf_counter()

    base_model = LinearSVC(
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=SVM_PARAM_GRID,
        cv=GRID_SEARCH_CV,
        scoring=GRID_SEARCH_SCORING,
        n_jobs=GRID_SEARCH_JOBS,
        verbose=1,
    )

    print("\nRunning Grid Search...\n")

    grid_search.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - start_time
    )

    print("\nBest Parameters")
    print("-" * 30)

    for key, value in grid_search.best_params_.items():
        print(f"{key:<10}: {value}")

    print(
        f"\nBest CV F1 Score : {grid_search.best_score_:.4f}"
    )

    print(
        f"\n✔ Completed in {training_time:.4f} seconds"
    )

    return (
        grid_search.best_estimator_,
        training_time,
    )

# ==========================================================
# Cross Validation
# ==========================================================

def cross_validate_models(
    nb_model,
    lr_model,
    svm_model,
    X_train,
    y_train,
):
    """
    Perform 5-Fold Stratified Cross Validation
    for all trained models.
    """

    print("\nCross Validation")
    print("-" * 40)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {

        "Naive Bayes": nb_model,

        "Logistic Regression": lr_model,

        "Optimized Linear SVM": svm_model,

    }

    report = []

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, model in models.items():

        scores = cross_validate(

            estimator=model,

            X=X_train,

            y=y_train,

            cv=cv,

            scoring=[
                "accuracy",
                "precision",
                "recall",
                "f1",
            ],

            n_jobs=-1,

        )

        accuracy = scores[
            "test_accuracy"
        ].mean()

        precision = scores[
            "test_precision"
        ].mean()

        recall = scores[
            "test_recall"
        ].mean()

        f1 = scores[
            "test_f1"
        ].mean()

        print(f"\n{name}")

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

        report.append(

            f"{name}\n"

            f"Accuracy : {accuracy:.4f}\n"

            f"Precision: {precision:.4f}\n"

            f"Recall   : {recall:.4f}\n"

            f"F1 Score : {f1:.4f}\n"

            + "-" * 40
            + "\n"

        )

    with open(
        CROSS_VALIDATION_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.writelines(report)

    print(
        "\n✔ Cross Validation report saved"
    )

    print(
        CROSS_VALIDATION_REPORT_FILE
    )
# ==========================================================
# Compare Models
# ==========================================================

def compare_models(
    nb_model,
    lr_model,
    svm_model,
    X_test,
    y_test,
):
    """
    Compare all trained models using accuracy.

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

    svm_accuracy = accuracy_score(
        y_test,
        svm_model.predict(X_test),
    )

    scores = {
        "Naive Bayes": nb_accuracy,
        "Logistic Regression": lr_accuracy,
        "Optimized Linear SVM": svm_accuracy,
    }

    print("\nModel Comparison")
    print("-" * 40)

    for name, score in scores.items():

        print(
            f"{name:<25} : {score:.4f}"
        )

    best_name = max(
        scores,
        key=scores.get,
    )

    best_model = {
        "Naive Bayes": nb_model,
        "Logistic Regression": lr_model,
        "Optimized Linear SVM": svm_model,
    }[best_name]

    print(f"\n✔ Best Model: {best_name}")

    return (
        best_model,
        best_name,
        scores,
    )

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

    svm_model, svm_time = train_linear_svm(
    X_train,
    y_train,
    )

    cross_validate_models(
    nb_model,
    lr_model,
    svm_model,
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
        svm_model,
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
    print(f"Linear SVM          : {svm_time:.4f} sec")