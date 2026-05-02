# Quora Duplicate Question Detection System

A Machine Learning based NLP application that detects whether two questions are duplicates or not, similar to the system used by Quora.  
The project combines **feature engineering, semantic embeddings, and machine learning** to determine question similarity.

---

## Project Overview

Quora receives millions of questions from users. Many of these questions are duplicates or semantically similar. Detecting duplicate questions helps improve content quality and user experience.

This project builds a **duplicate question detection system** using the **Quora Question Pairs dataset** and applies multiple NLP techniques to identify semantic similarity between question pairs.

---

## Features

- Text preprocessing and cleaning
- Feature engineering using multiple similarity techniques
- Fuzzy string matching
- Semantic similarity using **Sentence Transformers (SBERT)**
- Machine Learning model for classification
- FastAPI REST API for prediction
- Simple web interface for user input

---

## Tech Stack

**Programming Language**
- Python

**Libraries**
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- FuzzyWuzzy
- BeautifulSoup
- TensorFlow

**Backend Framework**
- FastAPI

**Deployment Tools**
- Uvicorn

---

## Dataset

The project uses the **Quora Question Pairs dataset**, which contains pairs of questions and a label indicating whether they are duplicates.

Dataset contains:

- Question 1
- Question 2
- Duplicate label (0 or 1)

Source:
https://www.kaggle.com/c/quora-question-pairs

---

## Feature Engineering

Several handcrafted and semantic features were used:

### Text Similarity Features
- Fuzzy ratio
- Partial fuzzy ratio
- Token sort ratio
- Token set ratio

### Length Based Features
- Absolute length difference
- Mean question length

### Word Matching Features
- First word match
- Last word match
- Word overlap ratio

### Character Similarity
- Character set overlap
- Character total counts

### Semantic Features
- SBERT cosine similarity
- SBERT dot product similarity
- SBERT absolute difference

### Substring Feature
- Longest common substring ratio

Total features used: **18**

---

## Model

The final trained model is saved using **Joblib** and used during inference.

Pipeline:
