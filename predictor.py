import numpy as np
import pandas as pd
from preprocess import preprocess
from features import fetch_token_features, fetch_length_features, fetch_fuzzy_features

def build_feature_dict(q1_raw: str, q2_raw: str, sbert) -> dict:
    q1 = preprocess(q1_raw)
    q2 = preprocess(q2_raw)
    row = pd.Series({"question1": q1, "question2": q2})
    feats = {}

    # Basic
    w1, w2 = set(q1.split()), set(q2.split())
    feats["word_share"] = len(w1 & w2) / (len(w1) + len(w2) + 1e-4)

    # Token features (8)
    tf = fetch_token_features(row)
    feats["cwc_min"]       = tf[0]
    feats["cwc_max"]       = tf[1]
    feats["csc_min"]       = tf[2]
    feats["csc_max"]       = tf[3]
    feats["ctc_min"]       = tf[4]
    feats["ctc_max"]       = tf[5]
    feats["last_word_eq"]  = tf[6]
    feats["first_word_eq"] = tf[7]

    # Length features (3)
    lf = fetch_length_features(row)
    feats["abs_len_diff"]        = lf[0]
    feats["mean_len"]            = lf[1]
    feats["longest_substr_ratio"]= lf[2]

    # Fuzzy features (4) — named explicitly, order doesn't matter now
    ff = fetch_fuzzy_features(row)
    feats["fuzz_ratio"]         = ff[0]
    feats["fuzz_partial_ratio"] = ff[1]
    feats["token_sort_ratio"]   = ff[2]
    feats["token_set_ratio"]    = ff[3]

    # SBERT features (3)
    e1 = sbert.encode([q1])
    e2 = sbert.encode([q2])
    e1n = e1 / (np.linalg.norm(e1) + 1e-10)
    e2n = e2 / (np.linalg.norm(e2) + 1e-10)
    feats["sbert_cosine"]  = float(np.sum(e1n * e2n))
    feats["sbert_absdiff"] = float(np.mean(np.abs(e1 - e2)))
    feats["sbert_dot"]     = float(np.sum(e1 * e2))

    return feats

def predict_pair(q1: str, q2: str, model, feature_cols: list, sbert):
    feats = build_feature_dict(q1, q2, sbert)
    # feature_cols from quora_features.pkl controls the exact order the model expects
    X = pd.DataFrame([feats])[feature_cols]
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    return pred, prob, feats