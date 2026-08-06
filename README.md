# 📩 SpamShield AI: Intelligent SMS Spam Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

### 🚀 AI-Powered SMS Spam Detection using Natural Language Processing & Machine Learning

Detect whether an SMS message is **Spam** or **Ham** using an end-to-end Machine Learning pipeline powered by **Natural Language Processing (NLP)** and **Scikit-Learn**.

Unlike many basic implementations, this project supports:

- Automatic multi-dataset loading
- Automatic dataset merging and duplicate removal
- Advanced NLP preprocessing pipeline
- Enhanced TF-IDF feature extraction (Bi-grams)
- Hyperparameter tuning using GridSearchCV
- Automatic model comparison
- Automatic best model selection
- 5-Fold Cross Validation
- ROC Curve generation
- Precision-Recall Curve generation
- Prediction confidence estimation
- Premium Flask Web Interface

The project is designed with a modular architecture, making it easy to extend with additional datasets and machine learning models.

</div>

---

# 📖 Project Overview

SMS spam messages are one of the most common forms of unwanted digital communication. This project builds an intelligent SMS Spam Detection System capable of classifying incoming text messages as either:

- ✅ Ham (Legitimate Message)
- 🚨 Spam (Unwanted Message)

The application performs complete text preprocessing, feature extraction using TF-IDF, model training, evaluation, and prediction through an interactive Flask web interface.

The project compares three supervised Machine Learning algorithms

- Multinomial Naive Bayes
- Best Selected Model (Optimized Linear SVM)
- Optimized Linear Support Vector Machine (Linear SVM)

The best-performing model is automatically selected using test accuracy after hyperparameter tuning.

---

# ✨ Features

✔ Automatic Multi-Dataset Loading

✔ Automatic Dataset Discovery

✔ Automatic Dataset Merging

✔ Duplicate SMS Removal

✔ Automatic Label Detection

✔ Robust CSV/Excel/JSON Loader

✔ NLP Text Cleaning

✔ Stopword Removal

✔ Porter Stemming

✔ Enhanced TF-IDF Feature Extraction

✔ Unigram + Bigram Features

✔ Train/Test Split

✔ Multinomial Naive Bayes

✔ Logistic Regression

✔ Optimized Linear SVM

✔ Hyperparameter Tuning (GridSearchCV)

✔ 5-Fold Cross Validation

✔ Automatic Best Model Selection

✔ Classification Report

✔ Confusion Matrix

✔ ROC Curve

✔ Precision-Recall Curve

✔ Prediction Confidence Score

✔ Evaluation Report

✔ Model Comparison Chart

✔ Interactive Flask Web Interface

✔ Responsive UI

✔ Character Counter

✔ Probability Bars

✔ Premium Dark Theme

✔ Modular Project Structure

✔ GitHub Ready

---

# 🧠 Machine Learning Pipeline

```
Dataset Folder
      │
      ▼
Scan All Datasets
      │
      ▼
Load & Merge Datasets
      │
      ▼
Remove Duplicate Records
      │
      ▼
Encode Labels
      │
      ▼
Text Preprocessing
      │
      ▼
Enhanced TF-IDF Feature Extraction
      │
      ▼
Train Machine Learning Models
      │
      ├──────────────► Multinomial Naive Bayes
      │
      ├──────────────► Logistic Regression
      │
      └──────────────► Optimized Linear SVM
                             │
                             ▼
                   Hyperparameter Tuning
                       (GridSearchCV)
                             │
                             ▼
                  5-Fold Cross Validation
                             │
                             ▼
                     Compare Model Performance
                             │
                             ▼
                 Select Best Performing Model
                             │
                             ▼
                     Save Trained Model
                             │
                             ▼
                 Model Performance Evaluation
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
   Confusion Matrix     ROC Curve     Precision-Recall Curve
            │                │                │
            └────────────────┼────────────────┘
                             ▼
                  Generate Evaluation Report
                             │
                             ▼
                    Predict New SMS Messages
```

---

# 📂 Project Structure

```
SMS-Spam-Classifier/
│
├── dataset/
│    │
│    ├── spam.csv
│    ├── combined_dataset.csv
│    ├── dataset2.csv
│    └── ...
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── experimentation.ipynb
│
├── outputs/
│      ├── confusion_matrix.png
│      ├── model_comparison.png
│      ├── evaluation_report.txt
│      ├── roc_curve.png
│      ├── precision_recall_curve.png
│      ├── cross_validation_report.txt
│      └── evaluation_report.txt
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── utils.py
│
├── static/
│   ├── css/
│   │      style.css
│   └── js/
│          script.js
│
├── templates/
│   └── index.html
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# 🛠 Tech Stack

### Programming Language

- Python 3.12

### Machine Learning

- Scikit-Learn

- GridSearchCV

- LinearSVM

- NumPy

- Pandas

### Natural Language Processing

- NLTK
- TF-IDF Vectorizer
- Porter Stemmer

### Visualization

- Matplotlib
- Seaborn
- WordCloud

### Backend

- Flask

### Frontend

- HTML5
- CSS3
- JavaScript

---

# 📊 Dataset

The project automatically scans the **dataset/** folder and loads every supported dataset.

Supported formats

- TSV
- CSV
- Excel (.xlsx)
- Excel (.xls)
- JSON

Features

- Automatic dataset discovery
- Automatic merging
- Duplicate removal
- Automatic column detection
- Robust CSV encoding detection

Simply place additional datasets inside the **dataset/** folder and retrain the model.

Target Labels

| Label | Meaning |
|--------|----------|
| 0      |   Ham    |
| 1      |   Spam   |

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/HimanshuS19/SMS-Spam-Classifier.git
```

Go to project folder

```bash
cd SMS-Spam-Classifier
```

Create virtual environment

```bash
python -m venv myenv
```

Activate environment

### Windows

```bash
myenv\Scripts\activate
```

### Linux / macOS

```bash
source myenv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Run the Flask application

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# 🚀 How to Use

### Step 1

Launch the Flask application.

```bash
python app.py
```

---

### Step 2

Open your browser and visit

```
http://127.0.0.1:5000
```

---

### Step 3

Enter an SMS message.

Example:

```
Congratulations!

You have won a FREE iPhone.

Click the link below to claim your reward.
```

Click

```
Analyze Message
```

The application will display

- Prediction (Spam / Ham)
- Confidence Score
- Spam Probability
- Ham Probability
- Processed Text



---
User

↓

Flask UI

↓

Prediction API

↓

TF-IDF Vectorizer

↓

Logistic Regression

↓

Prediction Result

# 🧪 Model Performance

Two Machine Learning algorithms were trained and compared.

|           Model         |   Accuracy |
|-------------------------|------------|
| Multinomial Naive Bayes | 96.77%     |
| Logistic Regression     | **94.01%** |
| Optimized Linear SVM    | **96.54%** |
The Logistic Regression model achieved the highest accuracy and was selected as the final prediction model.

---

# 📈 Evaluation Metrics

Final Logistic Regression Model

| Metric      | Score      |
|-------------|------------|
| Accuracy    | **96.95%** |
| Precision   | **96.15%** |
| Recall      | **89.85%** |
| F1 Score    | **92.22%** |
| ROC-AUC     | **98.85%** |
|Avg Precision| **97.22%** |

The model provides high precision for spam detection, reducing the likelihood of legitimate messages being incorrectly classified as spam.

---

# 📊 Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's predictions.

```
                 Predicted

              Ham      Spam

Actual Ham     TN        FP

Actual Spam    FN        TP
```

Generated automatically after training:

```
outputs/confusion_matrix.png
```

---

# 📉 Model Comparison

Both trained models are compared visually.

Generated automatically:

```
outputs/model_comparison.png
```

---

# 📄 Evaluation Report

A complete evaluation report is generated after training.

Location

```
outputs/evaluation_report.txt
```

Includes

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Average Precision
- Classification Report

---

# 💻 Command Line Prediction

Run

```bash
python predict.py
```

Example

```
Enter SMS:

Congratulations!

You've won a FREE iPhone.

Prediction

Spam

Confidence

74.22%
```

---

# 🌐 Web Application

The Flask web application provides an intuitive interface for real-time SMS classification.

Features include

- Modern Dark Theme
- Responsive Layout
- Animated Background
- Spam/Ham Probability Bars
- Confidence Score
- Processed Text Display
- Error Handling
- Character Counter

---

# 📸 Screenshots

## Home Page

```
assets/home.png
```

---

## Prediction Result

```
assets/result.png
```

*(Replace these placeholders with actual screenshots after deployment.)*

---

# 🔮 Future Improvements

Planned enhancements include:

• Random Forest
• XGBoost
• LightGBM
• CatBoost
• Deep Learning (LSTM)
• BERT / DistilBERT
• Explainable AI (SHAP)
• Docker Deployment
• REST API
• CI/CD using GitHub Actions
• MLflow Model Versioning
• ONNX Model Export
• Streamlit Dashboard

---

# 👨‍💻 Author

**Himanshu Singh**

B.Tech – Computer Science & Engineering (AI & ML)

JSS Academy of Technical Education, Noida

GitHub:

```
https://github.com/HimanshuS19
```

LinkedIn:

```
(Add your LinkedIn profile)
```

---

# 🙏 Acknowledgements

This project uses the following open-source resources:

- SMS Spam Collection Dataset
- Scikit-Learn
- Flask
- NLTK
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud

Special thanks to the open-source community for making these tools available.

---

# 📜 License

This project is intended for educational and research purposes.

You are free to use, modify, and extend this project with appropriate attribution.

---

# ⭐ Support

If you found this project useful,

please consider giving it a ⭐ on GitHub.

It helps support future development and encourages more open-source Machine Learning projects.

---

<div align="center">

## ⭐ Thank You for Visiting ⭐

Built with ❤️ using

**Python • Flask • Scikit-Learn • Enhanced TF-IDF • Optimized Linear SVM • Machine Learning**

</div>