
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
import numpy as np
import tensorflow as tf
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Quora Duplicate Question Checker")

# Load trained model
model = joblib.load("quora_best_model.pkl")

# Load SBERT model (TensorFlow backend)
tf.get_logger().setLevel('ERROR')  # suppress TF warnings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


# --- Pydantic schema for /predict endpoint ---
class QuestionPair(BaseModel):
    q1: str
    q2: str


# --- Preprocessing ---
def preprocess(q: str) -> str:
    q = str(q).lower().strip()
    q = q.replace('%', ' percent ').replace('$', ' dollar ').replace('₹', ' rupee ')
    q = q.replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '')
    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q).strip()
    return q


# --- Longest common substring ratio ---
def longest_substr_ratio(s1: str, s2: str) -> float:
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > longest:
                    longest = m[i][j]
            else:
                m[i][j] = 0
    return longest / max(len(s1), len(s2), 1)


# --- Feature extraction ---
def build_features(q1: str, q2: str) -> dict:
    words1, words2 = q1.split(), q2.split()
    len1, len2 = len(q1), len(q2)

    features = {}

    # Length-based features
    features['abs_len_diff'] = abs(len1 - len2)
    features['mean_len'] = (len1 + len2) / 2

    # First/last word match
    features['first_word_eq'] = int(words1[0] == words2[0]) if words1 and words2 else 0
    features['last_word_eq'] = int(words1[-1] == words2[-1]) if words1 and words2 else 0

    # Fuzzy features
    features['fuzz_ratio'] = fuzz.ratio(q1, q2)
    features['fuzz_partial_ratio'] = fuzz.partial_ratio(q1, q2)
    features['token_sort_ratio'] = fuzz.token_sort_ratio(q1, q2)
    features['token_set_ratio'] = fuzz.token_set_ratio(q1, q2)

    # Character/word set stats
    def char_word_stats(s1, s2):
        c1 = list(s1.replace(' ', ''))
        c2 = list(s2.replace(' ', ''))
        csc = [len(set(c1) & set(c2)) / max(1, len(set(c1) | set(c2)))]
        cwc = [len(set(words1) & set(words2)) / max(1, len(set(words1) | set(words2)))]
        ctc = [len(c1) + len(c2)]
        return csc, cwc, ctc

    csc, cwc, ctc = char_word_stats(q1, q2)
    features['csc_min'] = np.min(csc)
    features['csc_max'] = np.max(csc)
    features['cwc_min'] = np.min(cwc)
    features['cwc_max'] = np.max(cwc)
    features['ctc_min'] = np.min(ctc)
    features['ctc_max'] = np.max(ctc)

    # Longest substring ratio
    features['longest_substr_ratio'] = longest_substr_ratio(q1, q2)

    # SBERT embeddings
    emb1 = sbert_model.encode(q1, convert_to_numpy=True)
    emb2 = sbert_model.encode(q2, convert_to_numpy=True)
    features['sbert_cosine'] = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    features['sbert_dot'] = float(np.dot(emb1, emb2))
    features['sbert_absdiff'] = float(np.sum(np.abs(emb1 - emb2)))

    return features


# --- Feature order to match model ---
FEATURE_ORDER = [
    'cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
    'last_word_eq', 'first_word_eq', 'abs_len_diff', 'mean_len',
    'longest_substr_ratio', 'token_sort_ratio', 'token_set_ratio',
    'fuzz_ratio', 'fuzz_partial_ratio', 'sbert_cosine', 'sbert_absdiff', 'sbert_dot'
]

# --- HTML template (inline) ---
HTML_TEMPLATE = """
<!doctype html>
<html>
<head><title>Quora Question Pair Checker</title></head>
<body>
<h2>Quora Question Pair Duplicate Checker</h2>
<form method="post">
  Question 1:<br><input type="text" name="q1" size="60"><br><br>
  Question 2:<br><input type="text" name="q2" size="60"><br><br>
  <input type="submit" value="Check Duplicate">
</form>
{result_block}
</body>
</html>
"""


# --- Web interface (GET) ---
@app.get("/", response_class=HTMLResponse)
async def home_get():
    return HTML_TEMPLATE.format(result_block="")


# --- Web interface (POST) ---
@app.post("/", response_class=HTMLResponse)
async def home_post(q1: str = Form(""), q2: str = Form("")):
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    features = build_features(q1, q2)
    X_input = pd.DataFrame([features])[FEATURE_ORDER]
    pred = model.predict(X_input)[0]
    result = "Duplicate " if pred else "Not Duplicate "
    result_block = f"<h3>Result: {result}</h3>"
    return HTML_TEMPLATE.format(result_block=result_block)


# --- API endpoint ---
@app.post("/predict")
async def predict(payload: QuestionPair):
    try:
        q1 = preprocess(payload.q1)
        q2 = preprocess(payload.q2)
        features = build_features(q1, q2)
        X_input = pd.DataFrame([features])[FEATURE_ORDER]
        pred = model.predict(X_input)[0]
        return {"duplicate": bool(pred)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# --- Run with uvicorn ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)

import distance
from nltk.corpus import stopwords

# -------------------- Loading Artifacts --------------------
with open("quora_best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("quora_features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Loading SBERT
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Stopwords
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
def fetch_token_features(row):
    q1 = row['question1']
    q2 = row['question2']

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return [0.0]*8

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    return [
        common_word_count / (min(len(q1_words), len(q2_words)) + 1e-4),
        common_word_count / (max(len(q1_words), len(q2_words)) + 1e-4),
        common_stop_count / (min(len(q1_stops), len(q2_stops)) + 1e-4),
        common_stop_count / (max(len(q1_stops), len(q2_stops)) + 1e-4),
        common_token_count / (min(len(q1_tokens), len(q2_tokens)) + 1e-4),
        common_token_count / (max(len(q1_tokens), len(q2_tokens)) + 1e-4),
        int(q1_tokens[-1] == q2_tokens[-1]),
        int(q1_tokens[0] == q2_tokens[0]),
    ]


def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']

    return [
        abs(len(q1.split()) - len(q2.split())),
        (len(q1.split()) + len(q2.split())) / 2,
        len(distance.lcsubstrings(q1, q2)[0]) / (min(len(q1), len(q2)) + 1e-4)
        if distance.lcsubstrings(q1, q2) else 0
    ]


def fetch_fuzzy_features(row):
    q1 = row['question1']
    q2 = row['question2']

    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]


# -------------------- Prediction --------------------
def predict_duplicate(q1_raw, q2_raw):
    q1 = preprocess(q1_raw)
    q2 = preprocess(q2_raw)

    row = pd.Series({"question1": q1, "question2": q2})

    feats = {}

    # Token features
    tf = fetch_token_features(row)
    for col, val in zip([
        "cwc_min","cwc_max","csc_min","csc_max",
        "ctc_min","ctc_max","last_word_eq","first_word_eq"
    ], tf):
        feats[col] = val

    # Length features
    lf = fetch_length_features(row)
    for col, val in zip([
        "abs_len_diff","mean_len","longest_substr_ratio"
    ], lf):
        feats[col] = val

    # Fuzzy features
    ff = fetch_fuzzy_features(row)
    for col, val in zip([
        "fuzz_ratio","fuzz_partial_ratio",
        "token_sort_ratio","token_set_ratio"
    ], ff):
        feats[col] = val

    # SBERT
    e1 = sbert_model.encode([q1])
    e2 = sbert_model.encode([q2])

    e1n = e1 / np.linalg.norm(e1)
    e2n = e2 / np.linalg.norm(e2)

    feats["sbert_cosine"] = float(np.sum(e1n * e2n))
    feats["sbert_absdiff"] = float(np.mean(np.abs(e1 - e2)))
    feats["sbert_dot"] = float(np.sum(e1 * e2))

    X = pd.DataFrame([feats])[feature_cols]

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return pred, prob


# -------------------- API Endpoint --------------------
@app.post("/predict")
def predict(pair: QuestionPair):
    pred, prob = predict_duplicate(pair.question1, pair.question2)

    return {
        "question1": pair.question1,
        "question2": pair.question2,
        "is_duplicate": bool(pred),
        "probability": prob
    }
