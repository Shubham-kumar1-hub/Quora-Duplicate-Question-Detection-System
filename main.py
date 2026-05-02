import os, pickle
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from predictor import predict_pair

MODEL = None
FEATURE_COLS = None
SBERT = None

def load_artifacts():
    global MODEL, FEATURE_COLS, SBERT
    model_path    = os.getenv("MODEL_PATH",    "quora_best_model.pkl")
    features_path = os.getenv("FEATURES_PATH", "quora_features.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'. Train and save it first.")
    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)
    # Use the model's own feature names — guaranteed to match
    FEATURE_COLS = MODEL.get_booster().feature_names
    SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ All artifacts loaded.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield

app = FastAPI(
    title="Quora Duplicate Question Detection API",
    description="Detects duplicate Quora questions using NLP features (token, length, fuzzy) + SBERT embeddings + XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ────────────────────────────────────────────────────────────────────
class PairRequest(BaseModel):
    question1: str = Field(..., min_length=3, example="What is the best way to learn machine learning?")
    question2: str = Field(..., min_length=3, example="How can I start learning ML from scratch?")

class PredictResponse(BaseModel):
    question1: str
    question2: str
    is_duplicate: bool
    label: str
    duplicate_probability: float
    features: dict

class BatchRequest(BaseModel):
    pairs: list[PairRequest] = Field(..., max_length=50)

class BatchResponse(BaseModel):
    results: list[PredictResponse]

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "message": "Quora Duplicate Detection API 🚀", "docs": "/docs"}

@app.get("/health", tags=["Health"])
def health():
    return {
        "model_loaded":   MODEL is not None,
        "sbert_loaded":   SBERT is not None,
        "feature_count":  len(FEATURE_COLS) if FEATURE_COLS else 0,
    }

@app.get("/features", tags=["Info"])
def list_features():
    if not FEATURE_COLS:
        raise HTTPException(503, "Model not loaded")
    return {"feature_count": len(FEATURE_COLS), "features": FEATURE_COLS}

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PairRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    try:
        pred, prob, feats = predict_pair(req.question1, req.question2, MODEL, FEATURE_COLS, SBERT)
        return PredictResponse(
            question1=req.question1, question2=req.question2,
            is_duplicate=bool(pred),
            label="Duplicate" if pred else "Not Duplicate",
            duplicate_probability=round(prob, 4),
            features={k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()},
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(req: BatchRequest):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    results = []
    for p in req.pairs:
        pred, prob, feats = predict_pair(p.question1, p.question2, MODEL, FEATURE_COLS, SBERT)
        results.append(PredictResponse(
            question1=p.question1, question2=p.question2,
            is_duplicate=bool(pred),
            label="Duplicate" if pred else "Not Duplicate",
            duplicate_probability=round(prob, 4),
            features={k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()},
        ))
    return BatchResponse(results=results)