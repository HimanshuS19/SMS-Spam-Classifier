# 📩 SMS Spam Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange?style=for-the-badge&logo=scikit-learn)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

### 🚀 AI-Powered SMS Spam Detection using Natural Language Processing & Machine Learning

Detect whether an SMS message is **Spam** or **Ham** using a complete end-to-end Machine Learning pipeline built with **Python**, **Scikit-Learn**, and **Flask**.

</div>

---

# 📖 Project Overview

SMS spam messages are one of the most common forms of unwanted digital communication. This project builds an intelligent SMS Spam Detection System capable of classifying incoming text messages as either:

- ✅ Ham (Legitimate Message)
- 🚨 Spam (Unwanted Message)

The application performs complete text preprocessing, feature extraction using TF-IDF, model training, evaluation, and prediction through an interactive Flask web interface.

The project compares two classical Machine Learning algorithms:

- Multinomial Naive Bayes
- Logistic Regression

The best-performing model is automatically selected and saved for deployment.

---

# ✨ Features

✔ Complete NLP preprocessing pipeline

✔ Text Cleaning

✔ Stopword Removal

✔ Porter Stemming

✔ TF-IDF Vectorization

✔ Train/Test Split

✔ Multinomial Naive Bayes

✔ Logistic Regression

✔ Automatic Best Model Selection

✔ Model Performance Evaluation

✔ Classification Report

✔ Confusion Matrix

✔ Prediction Confidence

✔ Interactive Flask Web Interface

✔ Responsive Dark Theme UI

✔ GitHub Ready Project Structure

---

# 🧠 Machine Learning Pipeline

```
SMS Dataset
      │
      ▼
Load Dataset
      │
      ▼
Text Cleaning
      │
      ▼
Stopword Removal
      │
      ▼
Porter Stemming
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Train/Test Split
      │
      ▼
Naive Bayes
      │
      ▼
Logistic Regression
      │
      ▼
Model Comparison
      │
      ▼
Best Model Saved
      │
      ▼
Prediction
```

---

# 📂 Project Structure

```
SMS-Spam-Classifier/
│
├── dataset/
│   └── spam.csv
│
├── models/
│   ├── spam_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── experimentation.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── evaluation_report.txt
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
│   └── js/
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

**Dataset Used**

SMS Spam Collection Dataset

- Total Messages: **5,572**
- Ham Messages: **4,825**
- Spam Messages: **747**

Target Labels

| Label | Meaning |
|--------|----------|
| 0 | Ham |
| 1 | Spam |

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/SMS-Spam-Classifier.git
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

# 🧪 Model Performance

Two Machine Learning algorithms were trained and compared.

| Model | Accuracy |
|--------|----------|
| Multinomial Naive Bayes | 96.77% |
| Logistic Regression | **96.95%** |

The Logistic Regression model achieved the highest accuracy and was selected as the final prediction model.

---

# 📈 Evaluation Metrics

Final Logistic Regression Model

| Metric | Score |
|---------|--------|
| Accuracy | **96.95%** |
| Precision | **99.15%** |
| Recall | **77.85%** |
| F1 Score | **87.22%** |

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

- Merge larger modern SMS spam datasets (10K+ messages)
- Hyperparameter tuning using GridSearchCV
- Character-level TF-IDF features
- N-gram (Bigram/Trigram) support
- Support Vector Machine (LinearSVC)
- XGBoost classifier
- LightGBM classifier
- Deep Learning with LSTM
- Transformer-based models (BERT)
- Multilingual spam detection
- Email spam classification
- Explainable AI (SHAP/LIME)
- Cloud deployment with Docker and CI/CD

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

**Python • Flask • Scikit-Learn • NLP • Machine Learning**

</div>