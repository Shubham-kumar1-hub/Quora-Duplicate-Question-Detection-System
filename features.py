import nltk
import distance
import numpy as np
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
SAFE_DIV = 0.0001

def fetch_token_features(row):
    q1, q2 = row['question1'], row['question2']
    token_features = [0.0] * 8
    q1_tokens, q2_tokens = q1.split(), q2.split()
    if not q1_tokens or not q2_tokens:
        return token_features
    q1_words  = set(w for w in q1_tokens if w not in STOP_WORDS)
    q2_words  = set(w for w in q2_tokens if w not in STOP_WORDS)
    q1_stops  = set(w for w in q1_tokens if w in STOP_WORDS)
    q2_stops  = set(w for w in q2_tokens if w in STOP_WORDS)
    common_word  = len(q1_words  & q2_words)
    common_stop  = len(q1_stops  & q2_stops)
    common_token = len(set(q1_tokens) & set(q2_tokens))
    token_features[0] = common_word  / (min(len(q1_words),  len(q2_words))  + SAFE_DIV)
    token_features[1] = common_word  / (max(len(q1_words),  len(q2_words))  + SAFE_DIV)
    token_features[2] = common_stop  / (min(len(q1_stops),  len(q2_stops))  + SAFE_DIV)
    token_features[3] = common_stop  / (max(len(q1_stops),  len(q2_stops))  + SAFE_DIV)
    token_features[4] = common_token / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0]  == q2_tokens[0])
    return token_features

def fetch_length_features(row):
    q1, q2 = row['question1'], row['question2']
    q1_tokens, q2_tokens = q1.split(), q2.split()
    length_features = [0.0] * 3
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0.0
    return length_features

def fetch_fuzzy_features(row):
    q1, q2 = row['question1'], row['question2']
    # index 0=fuzz_ratio, 1=fuzz_partial_ratio, 2=token_sort_ratio, 3=token_set_ratio
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2),
    ]