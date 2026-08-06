"""
predict.py
----------
Prediction engine for the SpamShield AI: Intelligent SMS Spam Detection System.

Loads the trained model and TF-IDF vectorizer,
preprocesses incoming SMS text, and returns
prediction along with confidence scores.

Supports:
- Multinomial Naive Bayes
- Logistic Regression
- Optimized Linear SVM

Author: Himanshu Singh
Project: SpamShield AI: Intelligent SMS Spam Detection System
"""

from scipy.special import expit

from src.preprocessing import clean_text
from src.config import (
    MODEL_FILE,
    VECTORIZER_FILE,
)

from src.utils import (
    load_pickle,
    validate_input,
    format_percentage,
)


# ==========================================================
# Load Saved Objects
# ==========================================================

try:

    model = load_pickle(MODEL_FILE)

    vectorizer = load_pickle(
        VECTORIZER_FILE
    )

except Exception as e:

    raise RuntimeError(
        f"Unable to load trained model or vectorizer.\n{e}"
    )


# ==========================================================
# Prediction Function
# ==========================================================

def predict_sms(message: str) -> dict:
    """
    Predict whether an SMS is Spam or Ham.

    Parameters
    ----------
    message : str
        SMS entered by the user.

    Returns
    -------
    dict
        Prediction results.
    """

    if not isinstance(message, str):

        raise TypeError(
            "Input message must be a string."
        )

    message = validate_input(message)

    cleaned_text = clean_text(message)

    vector = vectorizer.transform(
        [cleaned_text]
    )

    prediction = int(
        model.predict(vector)[0]
    )

    # --------------------------------------------------
    # Models supporting probability estimates
    # (Naive Bayes / Logistic Regression)
    # --------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            vector
        )[0]

        ham_probability = float(
            probabilities[0]
        )

        spam_probability = float(
            probabilities[1]
        )

    # --------------------------------------------------
    # Linear SVM
    # Convert decision score into probability-like values
    # --------------------------------------------------

    elif hasattr(model, "decision_function"):

        score = float(
            model.decision_function(vector)[0]
        )

        spam_probability = float(
            expit(score)
        )

        ham_probability = (
            1.0 - spam_probability
        )

    # --------------------------------------------------
    # Fallback
    # --------------------------------------------------

    else:

        if prediction == 1:

            spam_probability = 1.0

            ham_probability = 0.0

        else:

            spam_probability = 0.0

            ham_probability = 1.0

    confidence = max(
        ham_probability,
        spam_probability,
    )

    label = (
        "Spam"
        if prediction == 1
        else "Ham"
    )

    return {

        "prediction": prediction,

        "label": label,

        "confidence": format_percentage(
            confidence * 100
        ),

        "ham_probability": format_percentage(
            ham_probability * 100
        ),

        "spam_probability": format_percentage(
            spam_probability * 100
        ),

        "clean_text": cleaned_text,

    }


# ==========================================================
# Module Testing
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SpamShield AI: Intelligent SMS Spam Detection System - Prediction")
    print("=" * 60)

    while True:

        message = input(
            "\nEnter SMS (type 'exit' to quit): "
        )

        if message.lower() == "exit":
            break

        try:

            result = predict_sms(
                message
            )

            print("\nPrediction Result")
            print("-" * 40)

            print(
                f"Prediction       : {result['label']}"
            )

            print(
                f"Confidence       : {result['confidence']}%"
            )

            print(
                f"Ham Probability  : {result['ham_probability']}%"
            )

            print(
                f"Spam Probability : {result['spam_probability']}%"
            )

            print(
                f"Processed Text   : {result['clean_text']}"
            )

        except Exception as e:

            print(f"\nError: {e}")

    print("\nProgram Closed.")