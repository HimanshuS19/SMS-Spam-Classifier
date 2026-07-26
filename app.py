"""
app.py
------
Flask web application for the SMS Spam Classifier.

Author: Himanshu Singh
Project: SMS Spam Classifier
"""

from flask import Flask, render_template, request

from predict import predict_sms

from src.config import (
    HOST,
    PORT,
    DEBUG,
    PROJECT_NAME,
)

# ==========================================================
# Initialize Flask App
# ==========================================================

app = Flask(__name__)


# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():
    """
    Display the home page.
    """
    return render_template(
        "index.html",
        project_name=PROJECT_NAME,
    )


# ==========================================================
# Prediction Route
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether the submitted SMS is Ham or Spam.
    """

    message = request.form.get("message", "").strip()

    if not message:
        return render_template(
            "index.html",
            project_name=PROJECT_NAME,
            error="Please enter an SMS message.",
        )

    try:

        result = predict_sms(message)

        return render_template(
            "index.html",
            project_name=PROJECT_NAME,
            message=message,
            prediction=result["label"],
            confidence=result["confidence"],
            ham_probability=result["ham_probability"],
            spam_probability=result["spam_probability"],
            cleaned_text=result["clean_text"],
        )

    except ValueError as e:

        return render_template(
            "index.html",
            project_name=PROJECT_NAME,
            message=message,
            error=str(e),
        )

    except Exception:

        return render_template(
            "index.html",
            project_name=PROJECT_NAME,
            message=message,
            error="An unexpected error occurred while processing your request.",
        )


# ==========================================================
# Run Application
# ==========================================================

if __name__ == "__main__":

    app.run(
        host=HOST,
        port=PORT,
        debug=DEBUG,
    )