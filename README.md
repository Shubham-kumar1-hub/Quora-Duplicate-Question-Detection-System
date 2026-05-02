# Quora Duplicate Question Detection — FastAPI

Detects whether two Quora questions are semantically duplicate using hand-crafted NLP features (token, length, fuzzy) + SBERT embeddings, classified with XGBoost.

---

## Project Structure
```
quora_fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app & all routes
│   ├── preprocess.py    # Text cleaning: abbreviations, contractions, HTML, punctuation
│   ├── features.py      # Token, length & fuzzy feature extraction
│   └── predictor.py     # Combines all features + SBERT + XGBoost inference
├── requirements.txt
└── README.md
```

---

## What's New (Updates)

### 1. Abbreviation Expansion
`preprocess.py` now expands common abbreviations **before** lowercasing so that
`"ML"` and `"machine learning"` are treated as the same tokens:

```
ML  → machine learning
AI  → artificial intelligence
NLP → natural language processing
DL  → deep learning
SQL → structured query language
API → application programming interface
... and more
```

### 2. Custom Decision Threshold
Instead of the default XGBoost threshold of `0.5`, `predictor.py` now uses `0.35`
for better recall — catching more real duplicates:

```python
THRESHOLD = 0.35  # tunable: lower = more duplicates caught
pred = 1 if prob >= THRESHOLD else 0
```

### 3. Feature Order Fix
Model feature names are now read directly from the booster itself instead of
`quora_features.pkl` (which was out of sync), guaranteeing correct column ordering:

```python
FEATURE_COLS = MODEL.get_booster().feature_names
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your trained model file in the project root
After running the notebook, copy this file here:
```
quora_best_model.pkl
```
> Note: `quora_features.pkl` is no longer needed — feature names are read directly from the model.

### 3. Run the server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open interactive API docs
```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Method | Endpoint         | Description                      |
|--------|------------------|----------------------------------|
| GET    | `/`              | Health check                     |
| GET    | `/health`        | Model & SBERT load status        |
| GET    | `/features`      | List all 18 model features       |
| POST   | `/predict`       | Predict a single question pair   |
| POST   | `/predict/batch` | Predict up to 50 pairs at once   |

---

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question1": "What is the best way to learn machine learning?",
    "question2": "How can I start learning ML from scratch?"
  }'
```

## Example Response
```json
{
  "question1": "What is the best way to learn machine learning?",
  "question2": "How can I start learning ML from scratch?",
  "is_duplicate": true,
  "label": "Duplicate",
  "duplicate_probability": 0.5823,
  "features": {
    "cwc_min": 0.9999,
    "cwc_max": 0.6667,
    "sbert_cosine": 0.8912,
    "fuzz_ratio": 74,
    "token_sort_ratio": 81,
    "..."
  }
}
```

---

## Features Used (18 total)

| Group | Features |
|---|---|
| Token-based (8) | cwc_min, cwc_max, csc_min, csc_max, ctc_min, ctc_max, last_word_eq, first_word_eq |
| Length-based (3) | abs_len_diff, mean_len, longest_substr_ratio |
| Fuzzy (4) | fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio |
| SBERT semantic (3) | sbert_cosine, sbert_absdiff, sbert_dot |

---

## Test Examples

### Should be Duplicate
```json
{"question1": "How do I lose weight fast?", "question2": "What is the quickest way to lose weight?"}
{"question1": "How do I learn ML?", "question2": "How can I learn machine learning?"}
{"question1": "What is the capital of India?", "question2": "Which city is the capital of India?"}
```

### Should NOT be Duplicate
```json
{"question1": "What is machine learning?", "question2": "How do I get a job in machine learning?"}
{"question1": "How do I start a business?", "question2": "How do I shut down my business?"}
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `feature_names mismatch` | Make sure `main.py` uses `MODEL.get_booster().feature_names` not the `.pkl` file |
| `attempted relative import` | Run `uvicorn main:app --reload` from the folder containing `main.py` |
| `Model not found` | Place `quora_best_model.pkl` in the same folder as `main.py` |
| `500 Internal Server Error` | Check terminal logs for the exact Python traceback |