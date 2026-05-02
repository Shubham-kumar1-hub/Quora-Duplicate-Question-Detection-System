from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import pandas as pd
import numpy as np
import pickle
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import distance
from nltk.corpus import stopwords

# -------------------- Load Artifacts --------------------
with open("quora_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ⚠️ feature_cols not used anymore (we enforce correct order manually)

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
STOP_WORDS = set(stopwords.words("english"))

# -------------------- FastAPI Init --------------------
app = FastAPI(title="Quora Duplicate Question Detector")

# -------------------- Request Schema --------------------
class QuestionPair(BaseModel):
    question1: str
    question2: str

# -------------------- Preprocessing --------------------
def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent ')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'[^a-z0-9\s]', '', q)
    return q

# -------------------- Feature Functions --------------------
def fetch_token_features(q1, q2):
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return [0.0]*8

    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])

    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])

    return [
        len(q1_words & q2_words) / (min(len(q1_words), len(q2_words)) + 1e-4),
        len(q1_words & q2_words) / (max(len(q1_words), len(q2_words)) + 1e-4),
        len(q1_stops & q2_stops) / (min(len(q1_stops), len(q2_stops)) + 1e-4),
        len(q1_stops & q2_stops) / (max(len(q1_stops), len(q2_stops)) + 1e-4),
        len(set(q1_tokens) & set(q2_tokens)) / (min(len(q1_tokens), len(q2_tokens)) + 1e-4),
        len(set(q1_tokens) & set(q2_tokens)) / (max(len(q1_tokens), len(q2_tokens)) + 1e-4),
        int(q1_tokens[-1] == q2_tokens[-1]),
        int(q1_tokens[0] == q2_tokens[0]),
    ]

def fetch_length_features(q1, q2):
    substrings = distance.lcsubstrings(q1, q2)
    longest = max((len(s) for s in substrings), default=0)

    return [
        abs(len(q1.split()) - len(q2.split())),
        (len(q1.split()) + len(q2.split())) / 2,
        longest / (min(len(q1), len(q2)) + 1e-4)
    ]

def fetch_fuzzy_features(q1, q2):
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]

# -------------------- Prediction Logic --------------------
def predict_duplicate(q1_raw, q2_raw):
    q1 = preprocess(q1_raw)
    q2 = preprocess(q2_raw)

    feats = {}

    # Token features
    tf = fetch_token_features(q1, q2)
    for col, val in zip([
        "cwc_min","cwc_max","csc_min","csc_max",
        "ctc_min","ctc_max","last_word_eq","first_word_eq"
    ], tf):
        feats[col] = val

    # Length features
    lf = fetch_length_features(q1, q2)
    for col, val in zip([
        "abs_len_diff","mean_len","longest_substr_ratio"
    ], lf):
        feats[col] = val

    # Fuzzy features (correct mapping)
    ff = fetch_fuzzy_features(q1, q2)
    feats["fuzz_ratio"] = ff[0]
    feats["fuzz_partial_ratio"] = ff[1]
    feats["token_sort_ratio"] = ff[2]
    feats["token_set_ratio"] = ff[3]

    # SBERT embeddings
    e1 = sbert_model.encode([q1])
    e2 = sbert_model.encode([q2])

    e1_norm = e1 / np.linalg.norm(e1)
    e2_norm = e2 / np.linalg.norm(e2)

    feats["sbert_cosine"] = float(np.sum(e1_norm * e2_norm))
    feats["sbert_absdiff"] = float(np.mean(np.abs(e1 - e2)))
    feats["sbert_dot"] = float(np.sum(e1 * e2))

    FEATURE_ORDER = [
        'cwc_min','cwc_max','csc_min','csc_max','ctc_min','ctc_max',
        'last_word_eq','first_word_eq','abs_len_diff','mean_len',
        'longest_substr_ratio',
        'token_sort_ratio','token_set_ratio',
        'fuzz_ratio','fuzz_partial_ratio',
        'sbert_cosine','sbert_absdiff','sbert_dot'
    ]

    X = pd.DataFrame([feats])[FEATURE_ORDER]

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return pred, prob

# -------------------- UI --------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Quora Duplicate Question Checker</h2>

            <form method="post">
                Question 1:<br>
                <input type="text" name="q1" size="60"><br><br>

                Question 2:<br>
                <input type="text" name="q2" size="60"><br><br>

                <input type="submit" value="Check">
            </form>
        </body>
    </html>
    """

@app.post("/", response_class=HTMLResponse)
def predict_form(q1: str = Form(...), q2: str = Form(...)):
    pred, prob = predict_duplicate(q1, q2)

    result = "Duplicate" if pred else "Not Duplicate"

    return f"""
    <html>
        <body>
            <h2>Result: {result}</h2>
            <p>Confidence: {round(prob*100, 2)}%</p>
            <a href="/">Try Again</a>
        </body>
    </html>
    """

# -------------------- API --------------------
@app.post("/predict")
def predict_api(pair: QuestionPair):
    pred, prob = predict_duplicate(pair.question1, pair.question2)

    return {
        "is_duplicate": bool(pred),
        "probability": prob
    }

# -------------------- Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)