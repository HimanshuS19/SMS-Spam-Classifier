"""
evaluation.py
-------------
Evaluates trained machine learning models for the
SMS Spam Classifier project.

Author: Himanshu Singh
Project: SMS Spam Classifier
"""

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import (
    OUTPUT_DIR,
    CONFUSION_MATRIX_FILE,
    MODEL_COMPARISON_FILE,
    EVALUATION_REPORT_FILE,
    LABEL_NAMES,
)


# ==========================================================
# Evaluate Model
# ==========================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.

    Parameters
    ----------
    model
        Trained machine learning model.

    X_test
        Testing feature matrix.

    y_test
        True labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=LABEL_NAMES,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ),
    }

    return metrics


# ==========================================================
# Print Metrics
# ==========================================================

def print_metrics(metrics):
    """
    Prints evaluation metrics to the console.
    """

    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    print("\nClassification Report\n")

    print(metrics["classification_report"])


# ==========================================================
# Save Evaluation Report
# ==========================================================

def save_evaluation_report(metrics):
    """
    Saves evaluation metrics into a text file.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        with open(
            EVALUATION_REPORT_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            file.write("SMS Spam Classifier Evaluation Report\n")
            file.write("=" * 45)
            file.write("\n\n")

            file.write(
                f"Accuracy : {metrics['accuracy']:.4f}\n"
            )

            file.write(
                f"Precision: {metrics['precision']:.4f}\n"
            )

            file.write(
                f"Recall   : {metrics['recall']:.4f}\n"
            )

            file.write(
                f"F1 Score : {metrics['f1_score']:.4f}\n\n"
            )

            file.write("Classification Report\n")
            file.write("-" * 45)
            file.write("\n")

            file.write(metrics["classification_report"])

        print("\n✔ Evaluation report saved")
        print(EVALUATION_REPORT_FILE)

    except Exception as e:
        raise RuntimeError(
            f"Unable to save evaluation report.\n{e}"
        )


# ==========================================================
# Plot Confusion Matrix
# ==========================================================

def plot_confusion_matrix(metrics):
    """
    Saves confusion matrix plot.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[
            "Predicted Ham",
            "Predicted Spam",
        ],
        yticklabels=[
            "Actual Ham",
            "Actual Spam",
        ],
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
    )

    plt.close()

    print("\n✔ Confusion Matrix saved")
    print(CONFUSION_MATRIX_FILE)


# ==========================================================
# Plot Model Comparison
# ==========================================================

def plot_model_comparison(scores):
    """
    Saves a bar chart comparing model accuracies.

    Parameters
    ----------
    scores : dict

    Example
    -------
    {
        "Naive Bayes":0.96,
        "Logistic Regression":0.97
    }
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_names = list(scores.keys())

    accuracies = list(scores.values())

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        model_names,
        accuracies,
    )

    plt.ylim(0.90, 1.00)

    plt.ylabel("Accuracy")

    plt.title("Model Accuracy Comparison")

    for bar, value in zip(bars, accuracies):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(
        MODEL_COMPARISON_FILE,
        dpi=300,
    )

    plt.close()

    print("\n✔ Model Comparison Chart saved")
    print(MODEL_COMPARISON_FILE)


# ==========================================================
# Complete Evaluation Pipeline
# ==========================================================

def evaluate_and_save(
    model,
    X_test,
    y_test,
    scores,
):
    """
    Runs the complete evaluation pipeline.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    print_metrics(metrics)

    save_evaluation_report(metrics)

    plot_confusion_matrix(metrics)

    plot_model_comparison(scores)

    return metrics


# ==========================================================
# Module Testing
# ==========================================================

if __name__ == "__main__":

    from src.preprocessing import preprocess_dataset

    from src.feature_engineering import (
        create_train_test_split,
        vectorize_data,
    )

    from src.model_training import (
        train_naive_bayes,
        train_logistic_regression,
        compare_models,
    )

    print("=" * 60)
    print("SMS Spam Classifier - Evaluation")
    print("=" * 60)

    # Load dataset
    df = preprocess_dataset()

    # Split data
    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    ) = create_train_test_split(df)

    # TF-IDF
    (
        X_train,
        X_test,
        vectorizer,
    ) = vectorize_data(
        X_train_text,
        X_test_text,
    )

    # Train models
    nb_model, _ = train_naive_bayes(
        X_train,
        y_train,
    )

    lr_model, _ = train_logistic_regression(
        X_train,
        y_train,
    )

    # Compare models
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

    print(f"\nBest Model Selected : {best_name}")

    # Complete evaluation
    evaluate_and_save(
        best_model,
        X_test,
        y_test,
        scores,
    )

    print("\n" + "=" * 60)
    print("Evaluation Completed Successfully")
    print("=" * 60)