# Quora Duplicate Question Detection — FastAPI

## Project Structure
```
quora_fastapi/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, routes
│   ├── preprocess.py    # Text cleaning (contractions, HTML, punctuation)
│   ├── features.py      # Token, length, fuzzy feature extraction
│   └── predictor.py     # Combines features + SBERT + model inference
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your trained model files in the project root
After running the notebook, copy these two files here:
- `quora_best_model.pkl`
- `quora_features.pkl`

### 3. Run the server
```bash
cd quora_fastapi
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open API docs
Visit: http://localhost:8000/docs

---

## API Endpoints

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/`               | Health check                       |
| GET    | `/health`         | Model/SBERT load status            |
| GET    | `/features`       | List all 18 model features         |
| POST   | `/predict`        | Predict a single question pair     |
| POST   | `/predict/batch`  | Predict up to 50 pairs at once     |

---

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
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
  "duplicate_probability": 0.8741,
  "features": {
    "cwc_min": 0.6667,
    "sbert_cosine": 0.9123,
    ...
  }
}
```

## Features Used (18 total)
- **Token-based (8):** cwc_min, cwc_max, csc_min, csc_max, ctc_min, ctc_max, last_word_eq, first_word_eq
- **Length-based (3):** abs_len_diff, mean_len, longest_substr_ratio
- **Fuzzy (4):** fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio
- **SBERT semantic (3):** sbert_cosine, sbert_absdiff, sbert_dot
