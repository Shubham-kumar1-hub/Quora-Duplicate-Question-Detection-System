import re
from bs4 import BeautifulSoup

CONTRACTIONS = {
    "ain't": "am not", "aren't": "are not", "can't": "can not",
    "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'll": "he will", "he's": "he is", "how'd": "how did",
    "how'll": "how will", "how's": "how is", "i'd": "i would",
    "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will",
    "it's": "it is", "let's": "let us", "might've": "might have",
    "mightn't": "might not", "must've": "must have", "mustn't": "must not",
    "needn't": "need not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "should've": "should have",
    "shouldn't": "should not", "that's": "that is", "there's": "there is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what's": "what is", "where's": "where is",
    "who's": "who is", "won't": "will not", "wouldn't": "would not",
    "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

ABBREV_MAP = {
    r'\bML\b': 'machine learning',
    r'\bAI\b': 'artificial intelligence',
    r'\bNLP\b': 'natural language processing',
    r'\bDL\b': 'deep learning',
    r'\bNN\b': 'neural network',
    r'\bCV\b': 'computer vision',
    r'\bDB\b': 'database',
    r'\bOS\b': 'operating system',
    r'\bAPI\b': 'application programming interface',
    r'\bSQL\b': 'structured query language',
}

def expand_abbreviations(q):
    for pattern, expansion in ABBREV_MAP.items():
        q = re.sub(pattern, expansion, q)
    return q

def preprocess(q):
    q = expand_abbreviations(str(q).strip())
    q = q.lower().strip()
    q = q.replace('%', ' percent ').replace('$', ' dollar ')
    q = q.replace('rupee', ' rupee ').replace('euro', ' euro ')
    q = q.replace('@', ' at ').replace('[math]', '')
    q = q.replace(',000,000,000 ', 'b ').replace(',000,000 ', 'm ').replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    q_decontracted = []
    for word in q.split():
        if word in CONTRACTIONS:
            word = CONTRACTIONS[word]
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have").replace("n't", " not")
    q = q.replace("'re", " are").replace("'ll", " will")
    q = BeautifulSoup(q, "html.parser").get_text()
    q = re.sub(r'\W', ' ', q).strip()
    return q