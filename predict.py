"""
predict.py
----------
Prediction engine for the SMS Spam Classifier.

Loads the trained model and TF-IDF vectorizer,
preprocesses incoming SMS text, and returns
prediction along with confidence scores.

Author: Himanshu Singh
Project: SMS Spam Classifier
"""

import joblib
import numpy as np

from src.preprocessing import clean_text

from src.config import (
    MODEL_FILE,
    VECTORIZER_FILE,
)


# ==========================================================
# Load Saved Objects
# ==========================================================

try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

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
        Input SMS message.

    Returns
    -------
    dict
        {
            prediction,
            label,
            confidence,
            spam_probability,
            ham_probability
        }
    """

    # ----------------------------------------------
    # Validate Input
    # ----------------------------------------------

    if not isinstance(message, str):
        raise TypeError("Input message must be a string.")

    message = message.strip()

    if not message:
        raise ValueError("Input message cannot be empty.")

    # ----------------------------------------------
    # Clean Text
    # ----------------------------------------------

    cleaned = clean_text(message)

    # ----------------------------------------------
    # Vectorize
    # ----------------------------------------------

    vector = vectorizer.transform([cleaned])

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    ham_probability = float(probabilities[0])

    spam_probability = float(probabilities[1])

    confidence = float(np.max(probabilities))

    label = "Spam" if prediction == 1 else "Ham"

    return {

        "prediction": int(prediction),

        "label": label,

        "confidence": round(confidence * 100, 2),

        "spam_probability": round(
            spam_probability * 100,
            2,
        ),

        "ham_probability": round(
            ham_probability * 100,
            2,
        ),

        "clean_text": cleaned,

    }


# ==========================================================
# Module Testing
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SMS Spam Classifier - Prediction")
    print("=" * 60)

    while True:

        message = input("\nEnter SMS (type 'exit' to quit): ")

        if message.lower() == "exit":
            break

        try:

            result = predict_sms(message)

            print("\nPrediction Result")
            print("-" * 40)

            print(f"Prediction       : {result['label']}")
            print(f"Confidence       : {result['confidence']}%")
            print(f"Ham Probability  : {result['ham_probability']}%")
            print(f"Spam Probability : {result['spam_probability']}%")
            print(f"Processed Text   : {result['clean_text']}")

        except Exception as e:
            print(f"\nError: {e}")

    print("\nProgram Closed.")