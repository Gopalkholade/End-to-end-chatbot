# End-to-End Intent-Based Chatbot

An intent-classification chatbot built with Python, **Scikit-Learn**, and **Streamlit**. The application uses TF-IDF vectorization and a Logistic Regression classifier to recognize user intent and return randomized, context-appropriate responses.

---

## 📋 Overview

This project demonstrates a lightweight, end-to-end NLP pipeline:
1. **Intent Dataset Creation (`test.py`)**: Defines structured conversational intents (e.g., `greeting`, `goodbye`, `thanks`, `about`, `help`, `age`, `weather`, `budget`, `credit_score`) and serializes them into `resources/intent.pkl`.
2. **Model Training & Inference (`main.py`)**: Vectorizes training patterns with `TfidfVectorizer`, trains a `LogisticRegression` model, exports the trained artifacts (`clf.pkl` and `vec.pkl`), and exposes the `chatbot()` prediction function.
3. **Interactive Web UI (`app.py`)**: Provides a browser-based chat interface built with Streamlit.

---

## 📂 Project Structure

```text
├── app.py                  # Streamlit web application entry point
├── main.py                 # Model training, serialization, and chatbot response logic
├── test.py                 # Script to generate and pickle the intents dataset
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
└── resources/
    ├── intent.pkl          # Serialized intent patterns, tags, and responses
    ├── clf.pkl             # Trained Logistic Regression classifier
    └── vec.pkl             # Fitted TF-IDF vectorizer
