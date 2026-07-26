"""
train.py
--------
Main training pipeline for the SMS Spam Classifier.

This script performs the complete workflow:

1. Load Dataset
2. Preprocess Data
3. Split Dataset
4. Convert Text using TF-IDF
5. Train Machine Learning Models
6. Compare Models
7. Save Best Model
8. Evaluate Best Model
9. Generate Reports

Author: Himanshu Singh
Project: SMS Spam Classifier
"""

from src.preprocessing import preprocess_dataset

from src.feature_engineering import (
    create_train_test_split,
    vectorize_data,
)

from src.model_training import (
    train_naive_bayes,
    train_logistic_regression,
    compare_models,
    save_model,
)

from src.evaluation import evaluate_and_save


# ==========================================================
# Main Training Pipeline
# ==========================================================

def main():
    """
    Execute the complete machine learning pipeline.
    """

    print("=" * 70)
    print("SMS Spam Classifier - Complete Training Pipeline")
    print("=" * 70)

    # ------------------------------------------------------
    # Step 1 : Load & Preprocess Dataset
    # ------------------------------------------------------

    print("\n[1/6] Loading Dataset...")

    dataframe = preprocess_dataset()

    print(f"✔ Dataset Loaded ({len(dataframe)} records)")

    # ------------------------------------------------------
    # Step 2 : Train/Test Split
    # ------------------------------------------------------

    print("\n[2/6] Splitting Dataset...")

    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    ) = create_train_test_split(dataframe)

    # ------------------------------------------------------
    # Step 3 : TF-IDF Vectorization
    # ------------------------------------------------------

    print("\n[3/6] Feature Engineering...")

    (
        X_train,
        X_test,
        vectorizer,
    ) = vectorize_data(
        X_train_text,
        X_test_text,
    )

    # ------------------------------------------------------
    # Step 4 : Train Models
    # ------------------------------------------------------

    print("\n[4/6] Training Models...")

    nb_model, nb_time = train_naive_bayes(
        X_train,
        y_train,
    )

    lr_model, lr_time = train_logistic_regression(
        X_train,
        y_train,
    )

    # ------------------------------------------------------
    # Step 5 : Compare Models
    # ------------------------------------------------------

    print("\n[5/6] Comparing Models...")

    (
        best_model,
        best_model_name,
        scores,
    ) = compare_models(
        nb_model,
        lr_model,
        X_test,
        y_test,
    )

    save_model(best_model)

    # ------------------------------------------------------
    # Step 6 : Evaluate Best Model
    # ------------------------------------------------------

    print("\n[6/6] Evaluating Best Model...")

    evaluate_and_save(
        best_model,
        X_test,
        y_test,
        scores,
    )

    print("\n" + "=" * 70)
    print("Training Pipeline Completed Successfully")
    print("=" * 70)

    print(f"\nBest Model        : {best_model_name}")
    print(f"Naive Bayes Time : {nb_time:.4f} sec")
    print(f"Logistic Reg.    : {lr_time:.4f} sec")

    print("\nGenerated Files")

    print("----------------------------------------")
    print("✔ models/spam_model.pkl")
    print("✔ models/tfidf_vectorizer.pkl")
    print("✔ outputs/evaluation_report.txt")
    print("✔ outputs/confusion_matrix.png")
    print("✔ outputs/model_comparison.png")


# ==========================================================
# Run Pipeline
# ==========================================================

if __name__ == "__main__":
    main()